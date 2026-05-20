/**
 * TranslationStatusDO — Durable Object for real-time pipeline progress.
 *
 * One instance per draft_id (keyed by name = String(draft_id)).
 *
 * Uses Hibernatable WebSockets:
 *   - The DO can hibernate between messages with zero cost.
 *   - Active connections are tracked by the runtime, not in-memory.
 *   - ctx.getWebSockets() returns all live connections after wake-up.
 *   - webSocketMessage / webSocketClose / webSocketError are the
 *     per-message callbacks — they receive the ws as parameter.
 *
 * Lifecycle:
 *   1. Client connects: GET /translations/:id/ws → proxy to this DO
 *   2. Pipeline stages call notify() via PipelineCallback RPC
 *   3. DO broadcasts to all connected clients via ctx.getWebSockets()
 *   4. On "done" or "error", last event is stored in SQLite storage
 *      so late-connecting clients get it immediately.
 *
 * Cost: only active while pipeline is running or clients are connected.
 */

import { DurableObject }         from "cloudflare:workers";
import type { Env, PipelineProgressEvent } from "../types";

const TERMINAL_KEY = "terminal_event";

export class TranslationStatusDO extends DurableObject<Env> {
  // ── WebSocket upgrade ───────────────────────────────────────────────

  override async fetch(req: Request): Promise<Response> {
    if (req.headers.get("Upgrade") !== "websocket") {
      return new Response("Expected WebSocket upgrade", { status: 426 });
    }

    const pair   = new WebSocketPair();
    const [client, server] = [pair[0], pair[1]];

    // acceptWebSocket enables Hibernatable WS — runtime tracks sessions,
    // not in-memory state. The DO can hibernate freely between messages.
    this.ctx.acceptWebSocket(server);

    // If pipeline already finished, send final event immediately so the
    // client doesn't have to wait. SQLite read is fast (< 1ms).
    const stored = await this.ctx.storage.get<PipelineProgressEvent>(TERMINAL_KEY);
    if (stored) {
      server.send(JSON.stringify(stored));
    }

    return new Response(null, { status: 101, webSocket: client });
  }

  // ── RPC called by PipelineCallback ─────────────────────────────────

  async notify(event: PipelineProgressEvent): Promise<void> {
    // Persist terminal events so late-joining clients get them immediately.
    if (event.type === "done" || event.type === "error") {
      await this.ctx.storage.put(TERMINAL_KEY, event);
    }

    const msg = JSON.stringify(event);

    // ctx.getWebSockets() returns ALL active sessions — works correctly
    // after hibernation (unlike an in-memory Set which would be empty).
    for (const ws of this.ctx.getWebSockets()) {
      try {
        ws.send(msg);
      } catch {
        // Client already closed — Hibernatable WS runtime will clean up.
      }
    }
  }

  // ── Hibernatable WS callbacks ───────────────────────────────────────
  // These replace the old webSocketClose/webSocketError overrides.
  // The runtime calls these after waking from hibernation.

  override webSocketClose(ws: WebSocket, code: number): void {
    ws.close(code);
  }

  override webSocketError(_ws: WebSocket, _err: unknown): void {
    // Nothing to do — the runtime removes the WS from ctx.getWebSockets()
    // automatically. No in-memory cleanup needed.
  }
}
