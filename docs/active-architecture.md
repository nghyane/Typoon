# Active architecture

Implementation rules live in [server-implementation-playbook.md](./server-implementation-playbook.md). If a server change conflicts with this file, follow the playbook and update this file if needed.

Current stack direction after Hono research:

- Runtime: Bun-native TypeScript.
- HTTP: Hono, served from `server/src/main.ts`, app composed in `server/src/app.ts` target layout.
- Validation: schema-first shared contracts with Valibot; Hono endpoints validate with `@hono/valibot-validator` and use `c.req.valid()` only.
- Contracts: explicit subpath imports only, for example `@typoon/contracts/translation/refine-window`. No root barrel export.
- Route typing: keep Hono handlers chained, do not annotate route factories as `Hono`, and export `AppType` from the composed app.
- Tests/smoke: use `app.request()`/`app.fetch()` and always set `Content-Type: application/json` for JSON validator tests.
- Cookies/session: use Hono cookie helpers, prefer `__Host-`/`httpOnly`/`secure`/`sameSite` session cookies for web sessions.
- OAuth: PWA session uses httpOnly cookie at the final origin. Cross-domain context is carried by server-side `flows`, not by cookies or URL payloads.
- LLM: modules call `GenerateText`; platform resolves purpose -> policy -> profile chain -> provider/protocol with fallbackable error classification.

Server layout rule:

```text
server/src/
  main.ts
  app.ts
  platform/                 # resources/adapters: env, db, tx, cookie, oauth, llm, payments
  modules/<domain>/         # routes.ts + domain-named business files
```

Route files only parse/validate HTTP input and call module functions with explicit deps. Business logic lives in domain-named module files. Avoid generic `services/`, `repositories/`, `utils/`, `helpers/`, `manager/` buckets.
