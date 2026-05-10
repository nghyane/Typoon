// Popup-driven picker + upload kickoff.
//
// Two phases run in different contexts:
//
//   * Pick (popup): inject content script, prompt user to pick the
//     image container, persist the selection to chrome.storage.
//
//   * Upload (background SW): fetch + pack + POST. Lives in the SW
//     so closing the popup or switching tabs doesn't abort the
//     pipeline. The popup observes progress via storage.onChanged
//     (see useUploadState).
//
// We keep `pickFromActiveTab`-style API for the popup but the
// heavy work is delegated to background.ts.

import type { ImageRef } from '@core/sources/extract'
import { TypoonClient } from '@core/typoon'
import { API_URL } from '@core/config'

const INJECT_FILE = 'content-scripts/content.js'
const PING_TIMEOUT_MS = 250

export interface PipelineProgress {
  phase: 'fetching' | 'packing' | 'uploading'
  cur:   number
  total: number
}

export interface PickedSelection {
  images:    ImageRef[]
  domain:    string
  sourceUrl: string
  /** Page <title> at pick time — used as a chapter-title hint in the form. */
  pageTitle: string
  tabId:     number
}

// ── 1. Pick images on the active tab ───────────────────────────────

const PENDING_PICK_KEY = 'typoon.pendingPick'

interface PendingPick {
  images:    ImageRef[]
  domain:    string
  sourceUrl: string
  pageTitle: string
  tabId:     number
  at:        number
}

/** Silent extract using the saved selector for this domain. Does
 *  NOT show the picker overlay — used on popup open to skip the
 *  click step when the domain is already trained. Returns `null`
 *  when the domain has no config or the selector no longer matches. */
export async function tryAutoPick(): Promise<PickedSelection | null> {
  const tab = await activeTab()
  if (!tab?.id || !tab.url) return null
  if (!isInjectableUrl(tab.url)) return null
  try {
    await ensureContentScript(tab.id)
  } catch {
    return null
  }
  const reply = (await browser.tabs.sendMessage(tab.id, { type: 'picker/auto' })) as
    { ok: boolean; images?: ImageRef[]; pageTitle?: string }
  if (!reply?.ok || !reply.images?.length) return null
  return {
    images:    reply.images,
    domain:    new URL(tab.url).host,
    sourceUrl: tab.url,
    pageTitle: reply.pageTitle ?? '',
    tabId:     tab.id,
  }
}

/** Read a buffered pick produced by the content script after the
 *  popup auto-closed (user clicked into the page). Clears the
 *  buffer after reading so subsequent popup opens see the empty
 *  state unless the user re-picks. */
export async function consumePendingPick(): Promise<PickedSelection | null> {
  const out = await browser.storage.local.get(PENDING_PICK_KEY)
  const raw = out[PENDING_PICK_KEY] as PendingPick | undefined
  if (!raw) return null
  await browser.storage.local.remove(PENDING_PICK_KEY)

  // Resolve tab: prefer the embedded id, fall back to active tab in
  // the user's current window. The match-by-domain check below is
  // the safety net — a stale pick from a different page should not
  // attempt to fetch from the wrong tab.
  let tabId = raw.tabId
  if (!tabId) {
    const tab = await activeTab()
    tabId = tab?.id ?? 0
  }
  if (!tabId) return null

  // Sanity: tab still open AND on the same origin we picked from.
  try {
    const tab = await browser.tabs.get(tabId)
    if (!tab?.url) return null
    if (new URL(tab.url).host !== raw.domain) return null
  } catch {
    return null
  }

  return {
    images:    raw.images,
    domain:    raw.domain,
    sourceUrl: raw.sourceUrl,
    pageTitle: raw.pageTitle,
    tabId,
  }
}

/** Mount the picker overlay on the active tab. Does NOT wait for the
 *  user to confirm — the popup typically loses focus and closes the
 *  moment the user clicks the page. The content script writes the
 *  result to chrome.storage.local; the next popup open reads it via
 *  `consumePendingPick`. */
export async function startPickerOnActiveTab(opts: { autoScroll?: boolean } = {}): Promise<void> {
  const tab = await activeTab()
  if (!tab?.id || !tab.url) throw new Error('Không tìm thấy tab đang mở.')
  if (!isInjectableUrl(tab.url)) {
    throw new Error('Trang này không hỗ trợ (chrome://, store, hoặc PDF).')
  }
  await ensureContentScript(tab.id)
  // Fire-and-forget. We don't await the reply — picker UX takes
  // unbounded time and popups die when focus moves to the tab.
  void browser.tabs.sendMessage(tab.id, {
    type: 'picker/activate',
    autoScroll: opts.autoScroll ?? false,
  }).catch(() => {})
}

// ── 2. Run the upload (fetch → pack → POST) ────────────────────────

// ── 2. Run the upload (delegates to the SW) ─────────────────────────

/** Enqueue a new upload job. The SW worker picks it up as soon as
 *  no other job is running; if the queue is empty it starts
 *  immediately. The popup observes progress via `useQueue`.
 *
 *  Returns the job id so the caller can correlate progress events
 *  back to a specific submission. */
export async function enqueueUpload(opts: {
  selection:  PickedSelection
  projectId:  number
  projectTitle?: string
  number?:    string
  title?:     string
}): Promise<string> {
  const reply = (await browser.runtime.sendMessage({
    type: 'queue/enqueue',
    job: {
      images:       opts.selection.images,
      projectId:    opts.projectId,
      projectTitle: opts.projectTitle,
      number:       opts.number,
      title:        opts.title,
      sourceUrl:    opts.selection.sourceUrl,
    },
  })) as { ok: boolean; id?: string; error?: string } | undefined

  if (!reply?.ok || !reply.id) {
    throw new Error(reply?.error ?? 'Không thêm được vào hàng đợi.')
  }
  return reply.id
}

/** Drop a finished/errored job from the queue (UI [×] button). */
export async function dismissJob(jobId: string): Promise<void> {
  await browser.runtime.sendMessage({ type: 'queue/dismiss', jobId }).catch(() => {})
}

/** Cancel a queued job before the worker picks it up. Running jobs
 *  cannot be cancelled mid-flight. */
export async function cancelJob(jobId: string): Promise<void> {
  await browser.runtime.sendMessage({ type: 'queue/cancel', jobId }).catch(() => {})
}

/** Re-enqueue a failed job. The job entry is reused so the user
 *  can see history of attempts; phase resets to `queued`. */
export async function retryJob(jobId: string): Promise<void> {
  await browser.runtime.sendMessage({ type: 'queue/retry', jobId }).catch(() => {})
}

// ── 3. Suggest the next chapter number ─────────────────────────────

/** Best-effort "what's the next chapter number?" — pulls existing
 *  chapters from the engine and returns `floor(maxNumeric) + 1` as
 *  a string. Falls back to '1' on error or empty list. */
export async function suggestNextNumber(
  token: string, projectId: number,
): Promise<string> {
  try {
    const client = new TypoonClient({ apiUrl: API_URL, token })
    const chapters = await client.listChapters(projectId)
    let max = 0
    for (const c of chapters) {
      const n = parseFloat(c.number)
      if (!isNaN(n) && n > max) max = n
    }
    return String(Math.floor(max) + 1)
  } catch {
    return ''
  }
}

// ── helpers ────────────────────────────────────────────────────────

async function activeTab(): Promise<Browser.tabs.Tab | undefined> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true })
  return tab
}

function isInjectableUrl(url: string): boolean {
  if (url.startsWith('chrome://'))             return false
  if (url.startsWith('chrome-extension://'))   return false
  if (url.startsWith('edge://'))               return false
  if (url.startsWith('about:'))                return false
  if (url.startsWith('view-source:'))          return false
  if (url.startsWith('https://chrome.google.com/webstore')) return false
  if (url.startsWith('https://chromewebstore.google.com'))  return false
  return true
}

async function pingTab(tabId: number): Promise<boolean> {
  try {
    const reply = await Promise.race([
      browser.tabs.sendMessage(tabId, { type: 'ping' }),
      new Promise<undefined>(r => setTimeout(() => r(undefined), PING_TIMEOUT_MS)),
    ])
    return Boolean((reply as { ok?: boolean } | undefined)?.ok)
  } catch {
    return false
  }
}

async function ensureContentScript(tabId: number): Promise<void> {
  if (await pingTab(tabId)) return
  console.log('[typoon] injecting content script into', tabId)
  await browser.scripting.executeScript({
    target: { tabId, allFrames: false },
    files: [INJECT_FILE],
  })
  for (let i = 0; i < 20; i++) {
    if (await pingTab(tabId)) return
    await new Promise(r => setTimeout(r, 50))
  }
  throw new Error('Content script không phản hồi sau khi inject.')
}
