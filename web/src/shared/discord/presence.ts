// Discord Activity SDK helpers — Rich Presence + orientation lock.
//
// All helpers are no-ops outside DA (DiscordSDKMock swallows them
// silently, but we double-guard with `isDiscordActivity` so we don't
// pay for SDK round trips when the user is just on the plain web app).
// Each command is idempotent and safe to call repeatedly.

import { Common } from '@discord/embedded-app-sdk'
import { discordSdk, isDiscordActivity } from './sdk'

// ── Rich Presence ──────────────────────────────────────────────────

export interface ReadingPresence {
  /** Project (manga/manhwa) title — appears as the bold line. */
  projectTitle: string
  /** Chapter number — appears as the second line ("Chapter 12"). */
  chapterNumber: string | number
  /** Optional chapter title appended after the number. */
  chapterTitle?: string | null
}

// Discord shows two lines under the app name:
//   line 1 = `details`  →  "Solo Leveling"
//   line 2 = `state`    →  "Chapter 12 — Awakening"
// Both lines are limited to 128 chars; over-length values are truncated
// at the boundary rather than rejected by the API.
function truncate(s: string, max = 128): string {
  return s.length <= max ? s : s.slice(0, max - 1) + '…'
}

/** Show "đang đọc <project> — Ch <n>" on the user's Discord profile. */
export async function setReadingPresence(p: ReadingPresence): Promise<void> {
  if (!isDiscordActivity) return
  const state = p.chapterTitle
    ? `Ch.${p.chapterNumber} — ${p.chapterTitle}`
    : `Ch.${p.chapterNumber}`
  try {
    await discordSdk.commands.setActivity({
      activity: {
        // Type 0 = "Playing". Discord renders DA presence this way.
        type: 0,
        details: truncate(p.projectTitle),
        state:   truncate(state),
      },
    })
  } catch (err) {
    // Don't crash the reader because Rich Presence failed (e.g. the
    // user's Discord client blocks the RPC scope). Log and move on.
    console.warn('[discord] setReadingPresence failed', err)
  }
}

/** Reset presence to the app's default (no details/state). */
export async function clearReadingPresence(): Promise<void> {
  if (!isDiscordActivity) return
  try {
    await discordSdk.commands.setActivity({
      activity: { type: 0, details: undefined, state: undefined },
    })
  } catch (err) {
    console.warn('[discord] clearReadingPresence failed', err)
  }
}

// ── Orientation lock ───────────────────────────────────────────────

type LockState = (typeof Common.OrientationLockStateTypeObject)[keyof typeof Common.OrientationLockStateTypeObject]

async function setOrientation(lock: LockState): Promise<void> {
  if (!isDiscordActivity) return
  try {
    await discordSdk.commands.setOrientationLockState({
      lock_state: lock,
      // PIP follows the same orientation policy as the focused view —
      // a reader pinned portrait should stay portrait when shrunk.
      picture_in_picture_lock_state: lock,
    })
  } catch (err) {
    console.warn('[discord] setOrientationLockState failed', err)
  }
}

/** Pin the Activity to portrait — webtoons are vertical strips. */
export const lockPortrait = () =>
  setOrientation(Common.OrientationLockStateTypeObject.PORTRAIT)

/** Restore default (UNLOCKED) — call when leaving the reader. */
export const unlockOrientation = () =>
  setOrientation(Common.OrientationLockStateTypeObject.UNLOCKED)
