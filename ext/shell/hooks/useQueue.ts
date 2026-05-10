// Subscribe to the SW-owned upload queue. Source of truth lives in
// chrome.storage.local under `typoon.queue`; the SW writes it as
// the worker advances and we mirror it into React state.
//
// Storage instead of runtime messages because the popup may close
// and re-open during a long upload — `storage.onChanged` fires
// regardless of whether anyone was listening at the time of the
// write, and `storage.local.get` gives us the current value on
// remount.

import { useEffect, useState } from 'react'
import {
  EMPTY_QUEUE, UPLOAD_QUEUE_KEY, type UploadQueue,
} from '@core/upload/state'
import { chromeStorage } from '@shell/adapters/chrome-storage'

export function useQueue(): UploadQueue {
  const [queue, setQueue] = useState<UploadQueue>(EMPTY_QUEUE)

  useEffect(() => {
    let alive = true
    void chromeStorage.get<UploadQueue>(UPLOAD_QUEUE_KEY).then(q => {
      if (alive && q) setQueue(q)
    })

    const onChange = (
      changes: Record<string, Browser.storage.StorageChange>,
      area:    Browser.storage.AreaName,
    ) => {
      if (area !== 'local' || !(UPLOAD_QUEUE_KEY in changes)) return
      const next = (changes[UPLOAD_QUEUE_KEY]?.newValue as UploadQueue | undefined) ?? EMPTY_QUEUE
      setQueue(next)
    }
    browser.storage.onChanged.addListener(onChange)
    return () => {
      alive = false
      browser.storage.onChanged.removeListener(onChange)
    }
  }, [])

  return queue
}
