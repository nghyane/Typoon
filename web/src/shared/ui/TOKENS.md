# Design tokens — Typoon SPA

Single source of truth for sizes, spacing, colors, and components.
Linear-inspired dense list/card hybrid. Read this before adding a new
surface or refactoring an existing one.

## Type scale

| Class       | px  | Use                                       |
|-------------|-----|-------------------------------------------|
| `text-xs`   | 12  | Meta, time, badges, sub-line              |
| `text-sm`   | 14  | Body default, table rows, buttons         |
| `text-base` | 16  | Card title emphasis                       |
| `text-lg`   | 18  | Page title                                |
| `text-2xl`  | 24  | Stat values (KPI)                         |

**Forbidden:** `text-[10px]`, `text-[11px]`, `text-[13px]`. Pick the
nearest scale. Custom sizes only for hero/landing — never inside
chrome or list rows.

## Icon sizes

| Size | Use                                  |
|------|--------------------------------------|
| 12   | Inline icon inside a text run        |
| 14   | Row action, table cell, button icon  |
| 16   | Sidebar nav, primary CTA             |
| 18   | BottomNav, mobile-only large tap     |
| 20   | Loading spinner (page-level)         |

## Heights

| Class    | px  | Use                                  |
|----------|-----|--------------------------------------|
| `h-6`    | 24  | Chips, pills, tags                   |
| `h-7`    | 28  | Secondary action, small input        |
| `h-8`    | 32  | Sidebar item, default button         |
| `h-9`    | 36  | Primary action, modal button         |
| `h-10`   | 40  | Large CTA                            |
| `h-bar`  | 44  | Top app bar                          |
| `h-14`   | 56  | List row card                        |

## Spacing scale

Use only: `gap-1` `gap-2` `gap-3` `gap-4` `gap-6`. Same for `p-*` and
`px-*`. `gap-1.5` allowed inside `Button` token only — not in surface
code.

## Border radius

| Class          | px   | Use                              |
|----------------|------|----------------------------------|
| `rounded-xs`   | 2    | Inputs, small chips              |
| `rounded-sm`   | 4    | Buttons, list items (default)    |
| `rounded-md`   | 6    | Cards, surface containers, modal |
| `rounded-full` | full | Pills, avatars                   |

## Surface hierarchy

| Token             | Use                                           |
|-------------------|-----------------------------------------------|
| `bg-bg`           | Page background                               |
| `bg-surface`      | Raised panels (sidebar, card, modal)          |
| `bg-surface-2`    | Recessed inside surface (input, table header) |
| `bg-hover`        | Interactive hover overlay                     |
| `bg-row-active`   | Selected row                                  |

## Status colors

| Token              | Meaning                                      |
|--------------------|----------------------------------------------|
| `accent` / `accent-text` / `accent-bg`   | Brand primary (active tab, CTA) |
| `info`   / `info-text`   / `info-bg`     | In-progress (running, spawn)    |
| `success`/ `success-text`/ `success-bg`  | Done, completed                 |
| `warning`/ `warning-text`/ `warning-bg`  | Pending, stale                  |
| `error`  / `error-text`  / `error-bg`    | Error, destructive              |
| `text` / `text-muted` / `text-subtle`    | Body / secondary / tertiary text |

## Components — when to use what

### Action surfaces

| Action                  | Component                                    |
|-------------------------|----------------------------------------------|
| Primary CTA (modal, FAB)| `<Button variant="primary">`                 |
| Secondary action        | `<Button variant="secondary">` (default)     |
| Toolbar / row inline    | `<Button variant="ghost" size="sm">`         |
| Icon-only ghost         | `<Button variant="ghost" size="sm" icon>`    |
| Destructive             | `<Button variant="danger">`                  |
| Link-style              | `<Link>` w/ `text-accent hover:underline`    |

**Never** write raw `<button>` for actions in surfaces. Always use
`Button` so hover / disabled / focus-ring stays consistent.

### Inputs

`<input className={input}>` from `shared/ui/primitives`. Pair with
`<Field label="…" hint="…">` for label + hint.

### Badges & tags

`Badge` for status with a dot (running / done / error). `Tag` for
static metadata (lang codes, "Chính thức", "NSFW"). Never write a
raw span pretending to be a chip.

### Card patterns

| Pattern        | Use                                          |
|----------------|----------------------------------------------|
| **Cover card** | Manga grid (feed, library grid). aspect-3/4, gradient overlay |
| **Row card**   | List items (more, library row, future settings). 56px row, icon-left + content + chevron-right |
| **Table row**  | Dense data (chapter list, queue, reports). 48px row. Status stripe left 2px |

### Empty state

`<EmptyState icon={Icon} title="…" hint="…">` always. Never a bare
centered paragraph.

### Loading

`<Spinner size={20}>` for page-level loading.
`<Spinner size={14}>` inline next to text.
`<Loader2 size={12} className="animate-spin">` for tiny inline (in a
table row sub-line).

## Transitions

| Use                | Class                                  |
|--------------------|----------------------------------------|
| Hover color/bg     | `transition-colors duration-150`       |
| Active state swap  | `transition-all duration-180`          |
| Tab / page swap    | Fade 200ms (router-level)              |
| Toolbar auto-hide  | `transition-transform duration-200`    |

Default: never `transition-all` unless animating multiple properties.

## Auditing

Run before merging a UI change:

```bash
# Forbidden ad-hoc text sizes
rg "text-\[1[0-3]px\]" web/src/

# Raw buttons that should be Button component
rg "<button[^>]*onClick" web/src/ | grep -v "Button"

# Custom gap sizes
rg "gap-1\.5" web/src/
```
