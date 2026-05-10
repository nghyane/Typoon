import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

// Same helper as web/src/shared/lib/cn.ts. Identical signature so UI
// snippets can be lifted between surfaces without rewrites.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
