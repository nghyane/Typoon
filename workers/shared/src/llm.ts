/**
 * LLM provider — packyapi OpenAI Responses API client.
 *
 * Key behaviour discovered while probing:
 *
 *   - Non-stream responses return `output: []` even when output_tokens > 0
 *     (proxy bug). Must use `stream: true` and accumulate
 *     `response.output_text.delta` deltas.
 *   - Setting `instructions` explicitly suppresses the proxy's default
 *     Codex CLI system preamble. Without it the model thinks it's a
 *     coding agent.
 *   - `reasoning.effort` defaults to "none" for `gpt-5.4-mini`, which is
 *     what we want for translation (deterministic, fast). We do not set
 *     it; if a future model defaults to "high" we'll add a knob.
 *
 * Public API:
 *   - `callLLM(env, system, user)`       — text-only (translate path).
 *   - `callLLMVision(env, system, parts)` — multimodal (brief path).
 *
 * Errors bubble; caller decides retry.
 */

export interface LLMEnv {
  PACKY_ENDPOINT: string;
  PACKY_MODEL:    string;
  PACKY_API_KEY:  string;          // wrangler secret
}

export interface LLMResult {
  text:        string;
  latency_ms:  number;
  usage?:      { input_tokens?: number; output_tokens?: number; total_tokens?: number };
}

/**
 * Subclass marker for errors that callers must treat as **fatal** —
 * authentication, quota, deliberate provider rejection. Mirrors the
 * `OperatorActionRequired` / `UpstreamUnavailable` typed exceptions in
 * `typoon/llm/errors.py`.
 *
 * Per-chunk callers in brief-worker run vision under `Promise.allSettled`
 * and degrade silently on transient failures, but a `LLMFatalError`
 * MUST be re-raised so the chapter fails loudly instead of silently
 * shipping an empty brief.
 */
export class LLMFatalError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "LLMFatalError";
  }
}

/** True for HTTP statuses that indicate a configuration / quota problem
 *  rather than a transient hiccup. */
function isFatalStatus(status: number): boolean {
  if (status === 401 || status === 403) return true;          // auth
  if (status === 402) return true;                            // billing
  if (status === 404) return true;                            // wrong endpoint
  if (status === 429) return true;                            // rate-limit / quota
  if (status >= 500 && status !== 503) return true;           // server bug, except temp unavailable
  return false;
}

/** Multimodal content part — text or inline image data URI. */
export type ContentPart =
  | { type: "input_text";  text: string }
  | { type: "input_image"; image_url: string };

export async function callLLM(
  env: LLMEnv, system: string, user: string,
): Promise<LLMResult> {
  return callLLMVision(env, system, [{ type: "input_text", text: user }]);
}

export async function callLLMVision(
  env: LLMEnv, system: string, parts: ContentPart[],
): Promise<LLMResult> {
  const t0 = Date.now();
  const body = {
    model: env.PACKY_MODEL,
    instructions: system,
    input: [{ role: "user", content: parts }],
    stream: true,
  };

  const resp = await fetch(env.PACKY_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${env.PACKY_API_KEY}`,
      "Accept":        "text/event-stream",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok || !resp.body) {
    const errBody = await resp.text().catch(() => "");
    const msg = `packy ${resp.status}: ${errBody.slice(0, 400)}`;
    if (isFatalStatus(resp.status)) {
      throw new LLMFatalError(resp.status, msg);
    }
    throw new Error(msg);
  }

  let text = "";
  let usage: LLMResult["usage"] | undefined;
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let lastEvent: string | null = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (line === "") { lastEvent = null; continue; }
      if (line.startsWith("event:")) {
        lastEvent = line.slice("event:".length).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        const payload = line.slice("data:".length).trim();
        if (!payload || payload === "[DONE]") continue;
        let obj: any;
        try { obj = JSON.parse(payload); } catch { continue; }
        const t = obj.type ?? lastEvent;
        if (t === "response.output_text.delta") {
          if (typeof obj.delta === "string") text += obj.delta;
        } else if (t === "response.completed") {
          const u = obj.response?.usage;
          if (u) usage = {
            input_tokens:  u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens:  u.total_tokens,
          };
        } else if (t === "response.error" || t === "error") {
          // Stream-level errors can carry the same auth / quota signal
          // as the HTTP status. Best-effort detection from the message.
          const payloadStr = JSON.stringify(obj).slice(0, 300);
          const low = payloadStr.toLowerCase();
          const fatal = low.includes("unauthor") || low.includes("forbidden")
            || low.includes("api key") || low.includes("invalid_api_key")
            || low.includes("billing") || low.includes("quota")
            || low.includes("permission");
          const msg = `packy stream error: ${payloadStr}`;
          throw fatal ? new LLMFatalError(0, msg) : new Error(msg);
        }
      }
    }
  }

  return { text, latency_ms: Date.now() - t0, usage };
}
