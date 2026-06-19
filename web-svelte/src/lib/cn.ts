// $lib/cn.ts — class name merger.  Filters falsy values, no runtime deps.
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ');
}
