// Content script. One responsibility: mount the picker overlay on
//
//   `picker/activate`  → show overlay, write the selection to
//                        chrome.storage so the popup can pick it up
//                        after it auto-closes.
//   `picker/auto`      → silent extract using the trained selector
//                        (no overlay), reply with the images.
//   `ping`             → liveness probe used during inject.
//
// Image fetch lives in the popup (extension context) which has
// `<all_urls>` host_permissions and therefore bypasses CDN CORS —
// content-script fetches would hit the gate.
//
// Listener is registered SYNCHRONOUSLY at module top-level so the
// popup's first sendMessage after `executeScript` always finds it.

import {
  extractImages, primeLazyLoad,
  type ImageRef,
} from '@core/sources/extract'
import { resolveSelector } from '@core/selectors/path'
import {
  getDomainConfig, saveDomainConfig, originKey,
} from '@core/selectors/domainConfig'
import { chromeStorage } from '@shell/adapters/chrome-storage'
import { mountPicker } from '@shell/picker/picker'

type Msg =
  | { type: 'ping' }
  | { type: 'picker/auto' }
  | { type: 'picker/activate'; autoScroll?: boolean }

interface PickerActivateReply {
  ok:     boolean
  images?: ImageRef[]
  /** Page title at pick time — popup uses it as a chapter title hint. */
  pageTitle?: string
  cancelled?: boolean
  error?: string
}

export default defineContentScript({
  matches: ['<all_urls>'],
  runAt: 'document_idle',
  main() {
    console.log('[typoon-cs] loaded on', location.host)
    // SYNCHRONOUS top-level register. Anything else (config load,
    // dynamic imports) runs inside the handler bodies.
    browser.runtime.onMessage.addListener((raw, _sender, respond) => {
      const msg = raw as Msg
      if (msg.type === 'ping') {
        respond({ ok: true })
        return false
      }
      if (msg.type === 'picker/auto') {
        handleAuto().then(respond, e =>
          respond({ ok: false, error: String(e) }),
        )
        return true
      }
      if (msg.type === 'picker/activate') {
        // Fire the picker overlay; reply immediately so the caller's
        // sendMessage settles. Result is buffered in chrome.storage
        // because the popup invariably closes before the user picks.
        respond({ ok: true })
        startPicker(msg.autoScroll ?? false)
        return false
      }
      return false
    })
  },
})

// ── handlers ───────────────────────────────────────────────────────

/** Silent extract using the trained selector. Used on popup open
 *  before showing any UI — if it succeeds, the popup goes straight
 *  to the upload form. No auto-scroll here: it's slow and user
 *  hasn't asked for it yet. */
async function handleAuto(): Promise<PickerActivateReply> {
  const domain = originKey(location.href)
  const config = await getDomainConfig(chromeStorage, domain)
  if (!config) return { ok: false }
  const el = resolveSelector(config.imagesSelector)
  if (!el) return { ok: false }
  await primeLazyLoad(el)
  const images = extractImages(el)
  if (images.length === 0) return { ok: false }
  return { ok: true, images, pageTitle: document.title }
}

/** Mount the picker, save the picked selection to chrome.storage on
 *  confirm, and notify any open popup. The popup almost always
 *  closes when the user clicks into the page, so the storage write
 *  is the primary handoff path; the runtime broadcast is a best-
 *  effort live update for the rare detached-popup case. */
function startPicker(autoScroll: boolean) {
  const domain = originKey(location.href)
  console.log('[typoon-cs] activate on', domain, 'autoScroll:', autoScroll)
  void getDomainConfig(chromeStorage, domain).then(config => {
    mountPicker({
      hint: config?.imagesSelector,
      defaultAutoScroll: autoScroll,
      onCancel: () => {
        // Nothing to persist — the next popup open just shows the
        // existing form (or empty state if there was nothing).
      },
      onConfirm: async (r) => {
        await saveDomainConfig(chromeStorage, {
          domain,
          imagesSelector: r.selector,
          expectedCount:  r.expectedCount,
          createdAt:      Date.now(),
        })
        const tabId = await currentTabId()
        await chromeStorage.set('typoon.pendingPick', {
          images:    r.images,
          domain,
          sourceUrl: location.href,
          pageTitle: document.title,
          tabId,
          at:        Date.now(),
        })
      },
    })
  })
}

/** Best-effort: the content script can't ask its own tab id directly,
 *  but the SW can. The popup, when it next opens, looks up the tab
 *  id of the active tab anyway — but the user might switch tabs
 *  between picking and re-opening the popup, so we prefer to embed
 *  the actual tab id at pick time. Falls back to 0 (which the popup
 *  will treat as "use active tab" when reading). */
async function currentTabId(): Promise<number> {
  try {
    // browser.runtime.sendMessage with sender.tab — the SW reflects
    // the sender back. We add a `tab/whoami` echo handler in the SW
    // for this. Falls back to active-tab lookup in the popup.
    const r = (await browser.runtime.sendMessage({ type: 'tab/whoami' })) as
      { ok?: boolean; tabId?: number } | undefined
    return r?.tabId ?? 0
  } catch {
    return 0
  }
}
