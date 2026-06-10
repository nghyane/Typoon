# Server implementation playbook

This is the coding contract for the Typoon server. Read this before adding or
changing server code. The goal is not enterprise ceremony; the goal is code that
is easy to read, hard to misuse, and safe around xu/payment logic.

## 0. Scope

Use a personal-project-grade modular monolith:

- typed Hono routes
- schema-first contracts
- plain functions
- explicit dependencies
- Result for expected business errors
- immutable wallet ledger for xu
- admin-configurable LLM provider/profile/policy, without a heavy framework

Do not add enterprise-only layers unless the user asks or production evidence
requires them.

## 1. Target layout

```text
server/src/
  main.ts
  app.ts

  platform/
    env.ts
    http-result.ts
    db.ts              # when DB code starts
    tx.ts              # when DB code starts
    crypto.ts          # when auth/payment needs it
    session.ts         # PWA cookie session

    oauth/
      discord.ts

    payments/
      payos.ts

    llm/
      types.ts
      client.ts
      errors.ts
      protocols/
        openai-responses.ts
        openai-chat.ts
        anthropic-messages.ts

  modules/
    translation/
      routes.ts
      session.ts
      refine-window.ts
      prompt.ts
      parser.ts
      prompt-key.ts
      cache-key.ts

    wallet/
      routes.ts
      ledger.ts
      queries.ts

    payment/
      routes.ts
      payos-webhook.ts

    auth/
      routes.ts
      flow.ts

    admin-llm/
      routes.ts
```

Contracts live in explicit subpaths:

```text
packages/contracts/src/
  common/error.ts
  translation/refine-window.ts
  translation/session.ts
  llm/provider.ts
  llm/profile.ts
  llm/policy.ts
  wallet/wallet.ts
  wallet/ledger.ts
  payment/order.ts
```

No root barrel export for contracts.

## 2. Core code pattern

Every HTTP use-case follows this path:

```text
routes.ts -> validate -> command/query function -> AppResult -> respond()
```

### Route pattern

Routes are HTTP boundary only.

```ts
export function translationRoutes(deps: TranslationDeps) {
  return new Hono()
    .post(
      '/api/translation-sessions/:id/refine-windows',
      vValidator('json', RefineWindowRequestSchema, invalidRequest),
      async c => {
        const input = c.req.valid('json')
        const sessionId = c.req.param('id')

        return respond(c, await refineWindow({ sessionId, input }, deps), 200)
      },
    )
}
```

Rules:

- Do not annotate route factories as `: Hono`; keep route type detail.
- Do not call `await c.req.json<T>()`.
- Do not put business logic in route handlers.
- Do not call external providers directly from routes.

### Business function pattern

Business functions are plain async functions and return `AppResult` for expected
errors.

```ts
export async function refineWindow(
  command: RefineWindowCommand,
  deps: RefineWindowDeps,
): Promise<AppResult<RefineWindowResponse>> {
  if (command.sessionId !== command.input.sessionId) {
    return err(400, 'session_id_mismatch', 'sessionId mismatch')
  }

  if (!command.input.activeUnitIds.length) {
    return err(422, 'active_units_required', 'Active unit ids are required')
  }

  return ok(response)
}
```

Rules:

- Business code must not import Hono.
- Business code must not import `env`.
- Business code must not know provider/protocol names unless it is inside
  `platform/`.
- Throw only for unexpected programmer bugs. Expected business failures return
  `err(...)`.

## 3. Result and response pattern

Use one shared result shape for expected outcomes:

```ts
export type AppResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: AppError }

export interface AppError {
  readonly status: 400 | 401 | 403 | 404 | 409 | 422 | 500 | 502
  readonly code: string
  readonly message: string
}
```

Endpoint response is centralized:

```ts
return respond(c, result, 200)
```

Stable error response:

```json
{
  "error": {
    "code": "session_id_mismatch",
    "message": "sessionId mismatch"
  }
}
```

Rules:

- No `instanceof` for expected business errors.
- No class-based service errors for normal validation/state problems.
- Error `code` is stable snake_case; message can change.

## 4. Contract pattern

Contracts are schema-first with Valibot.

```ts
export const RefineWindowRequestSchema = v.object({
  sessionId: v.pipe(v.string(), v.nonEmpty()),
  targetLang: v.pipe(v.string(), v.nonEmpty()),
})

export type RefineWindowRequest = v.InferOutput<typeof RefineWindowRequestSchema>
```

Rules:

- Do not write an interface beside a schema for the same DTO.
- Do not export from `@typoon/contracts` root.
- Use explicit imports:

```ts
import { RefineWindowRequestSchema } from '@typoon/contracts/translation/refine-window'
```

## 5. Dependency pattern

Pass only what a use-case needs.

Good:

```ts
export interface RefineWindowDeps {
  readonly generateText: GenerateText
}
```

Avoid:

```ts
export interface Deps {
  readonly env: ServerEnv
  readonly config: Config
  readonly services: ServiceContainer
}
```

Rules:

- No dependency container.
- No passing full env/config into business functions.
- DB dependencies should be explicit: `tx`, `now`, `newId`, specific read/write
  functions.

## 6. LLM pattern

Feature modules call a purpose-level text generator:

```ts
generateTextForPurpose('translation_refined', input)
```

LLM platform resolves:

```text
purpose -> policy -> profile chain -> provider -> protocol adapter
```

Concepts:

- provider: endpoint/key owner, e.g. OpenAI, OpenRouter, Anthropic, local
- profile: concrete model + protocol + endpoint path + params
- protocol: request/response shape, e.g. `openai_responses`,
  `openai_chat_completions`, `anthropic_messages`
- policy: ordered profile chain for a purpose

Rules:

- Do not name profiles after use-cases like `refine-default`.
- Do not infer protocol from provider.
- Translation must not import protocol adapters.
- Fallback only for retryable/provider failures. Do not fallback for config/auth
  errors, invalid prompt, or unsupported protocol.

## 7. Wallet/xu/payment rules

Money code has stricter rules than normal feature code.

Required tables/concepts:

```text
wallet_accounts
wallet_ledger
coin_packages
payment_orders
payment_webhook_events
translation_holds
translation_entitlements
```

Ledger kinds:

```text
topup
hold
capture
release
refund
admin_adjustment
```

Rules:

- Never expose an endpoint that sets wallet balance directly.
- Every xu mutation appends `wallet_ledger` and updates `wallet_accounts` in the
  same DB transaction.
- DB constraints must keep `available_xu >= 0` and `held_xu >= 0`.
- PayOS webhook must verify signature before processing money.
- PayOS webhook must be idempotent; duplicate webhook must not top up twice.
- Translation entitlement prevents paying twice for the same content key and
  target language.
- Hold/capture/release must be idempotent where retries can happen.

## 8. Auth/PWA rules

Primary auth is PWA-first:

```text
Discord OAuth callback -> create auth session -> set httpOnly secure cookie
```

Cross-domain context is stored server-side in `flows`, not in cookie payloads or
long URL payloads.

Rules:

- Cookie is session transport, not business context storage.
- Session token is opaque and stored hashed in DB.
- Do not add bearer/device-code flow until the user asks or a real non-PWA
  client requires it.

## 9. Forbidden names and structures

Do not create these folders/files for new code:

```text
services/
repositories/
utils/
helpers/
manager/
processor/
engine/
```

Use names that describe the business action:

```text
refine-window.ts
ledger.ts
payos-webhook.ts
session.ts
flow.ts
prompt.ts
parser.ts
```

## 10. New use-case checklist

Before coding a new use-case:

1. Pick the module: `translation`, `wallet`, `payment`, `auth`, `admin-llm`.
2. Add or reuse a contract schema in `packages/contracts/src/<module>/...`.
3. Add a route in `modules/<module>/routes.ts`.
4. Add a plain function for business logic in a domain-named file.
5. Return `AppResult` for expected failures.
6. Pass explicit deps only.
7. If money changes, use transaction + ledger.
8. If external system is involved, put protocol code under `platform/`.
9. Run the smallest meaningful verification.

## 11. Verification checklist

After server changes, run at least:

```text
bun run --cwd packages/contracts typecheck
bun run --cwd server typecheck
```

For route changes, run an `app.request()` smoke with JSON `Content-Type`.

For client/shared changes, also run:

```text
bun run --cwd packages/client typecheck
```

Do not say “should work”; include the actual command output summary.

## 12. Migration rule for current skeleton

Current code may still have transitional paths. When touching server code:

- move touched translation code toward `modules/translation`
- move `http.ts` toward `app.ts`
- do not add new code under `features/`
- do not keep compatibility wrappers after the new path verifies

Small diffs are preferred. Change one layer at a time.
