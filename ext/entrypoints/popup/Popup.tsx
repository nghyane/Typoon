// Popup root view-machine. Two views: Setup (no token yet) or
// Import (token saved). The Import flow handles its own internal
// states (idle/picking/uploading/done/error) — see ImportView.

import { useConfig } from '@shell/hooks/useConfig'
import { SetupView } from './views/SetupView'
import { ImportView } from './views/ImportView'

export function Popup() {
  const { ready, authed } = useConfig()

  if (!ready) {
    return (
      <div className="w-[360px] p-4 text-xs text-text-subtle">Đang tải…</div>
    )
  }
  return authed ? <ImportView /> : <SetupView />
}
