// InlineConfirm — replaces native window.confirm() inside the popup.
//
// Why: Chrome popups close the moment they lose focus. Opening a
// modal alert from the popup ships focus to the alert and the popup
// dismisses; the user clicks OK on the alert, but the original
// action context (the popup) is gone. Inline confirm keeps focus
// inside the popup the whole time.
//
// Pattern (GitHub destructive actions): two-stage button. First
// click switches the label and tone to a strong "Confirm?" prompt;
// second click within `holdMs` runs the action. A timeout reverts
// the prompt if the user walks away.

import { useEffect, useRef, useState, type ButtonHTMLAttributes } from 'react'
import { cn } from '@shared/lib/cn'

interface Props extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> {
  /** First-stage label. */
  label:        string
  /** Confirm-stage label. Defaults to "Bấm lần nữa". */
  confirmLabel?: string
  /** Time the confirm window stays open before reverting. */
  holdMs?:      number
  onConfirm:    () => void
  /** Tone classes for primed state. Defaults to error red. */
  primedClass?: string
}

export function InlineConfirm({
  label, confirmLabel = 'Bấm lần nữa',
  holdMs = 3000, onConfirm,
  primedClass = 'text-error-text font-semibold',
  className, ...rest
}: Props) {
  const [primed, setPrimed] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [])

  function arm() {
    setPrimed(true)
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => setPrimed(false), holdMs)
  }

  function fire() {
    if (timer.current) { clearTimeout(timer.current); timer.current = null }
    setPrimed(false)
    onConfirm()
  }

  return (
    <button
      type="button"
      onClick={primed ? fire : arm}
      className={cn(
        'underline-offset-2 hover:underline',
        primed ? primedClass : 'text-text-subtle hover:text-text-muted',
        className,
      )}
      {...rest}
    >
      {primed ? confirmLabel : label}
    </button>
  )
}
