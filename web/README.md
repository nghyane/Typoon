# Typoon Web

React 19 + Vite + Tailwind v4 + TanStack Router/Query + Zustand.

## Setup

```bash
export GITHUB_TOKEN=$(gh auth token)   # one-time, or add to ~/.zshrc
bun install
bun dev
```

`@nghyane/bunle` is hosted on GitHub Packages, so `GITHUB_TOKEN` (any
scope including `read:packages`) is required for `bun install`.

## Scripts

| | |
|---|---|
| `bun dev`     | Vite dev server, proxies `/api` and `/files` to `http://localhost:8000` |
| `bun build`   | `tsc -b && vite build` — typecheck + production bundle |
| `bun lint`    | ESLint |
| `bun preview` | Serve the built `dist/` |

Override the public origin with `VITE_PUBLIC_BASE_URL=...` (e.g. for
cross-origin preview deploys or pointing at a staging DA host).

## Layout

```
src/
  app/          application shell — layout, sidebar, header
  shared/       reusable primitives (no feature knowledge)
    ui/         Button, Modal, Toaster, Cover, primitives, …
    lib/        cn, time, events, useDelayedFlag
    api/        api.ts — fetch client + types (single source of truth)
  features/     feature-scoped code, single consumer
    auth/
    projects-list/
    project-detail/
    chapter-reader/
  routes/       thin route components, compose features
```

Path aliases: `@app/*`, `@shared/*`, `@features/*` (see `vite.config.ts`).
