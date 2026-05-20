/**
 * PipelineCallback — WorkerEntrypoint RPC called by typoon-pipeline.
 *
 * The pipeline's final step calls:
 *   env.API.finalize(args)           → mark draft done, write D1
 *   env.API.notifyProgress(args)     → forward progress event to DO WebSocket
 *
 * Declared as WorkerEntrypoint so it can receive Service Binding RPC.
 * Mounted in index.ts alongside the Hono fetch handler.
 */

import { WorkerEntrypoint } from "cloudflare:workers";
import type { Env, FinalizeArgs, PipelineProgressEvent, PipelineStage } from "../types";

export class PipelineCallback extends WorkerEntrypoint<Env> {
  /**
   * Called by pipeline finalize step after typeset-pack succeeds.
   * Atomically marks draft as done and stores scan/mask prefixes on chapter.
   */
  async finalize(args: FinalizeArgs): Promise<void> {
    const {
      chapter_id, draft_id, archive_key,
      page_count, scan_keys, mask_keys,
    } = args;

    // Derive R2 prefixes from the first key (all keys share the same prefix)
    const prepared_prefix = `prepared/${chapter_id}/`;
    const masks_prefix    = `mask/${chapter_id}/`;

    await this.env.DB.batch([
      // Mark draft done
      this.env.DB.prepare(
        `UPDATE translation_drafts
         SET state = 'done', archive_key = ?, rendered_at = datetime('now'),
             progress_stage = 'finalize', updated_at = datetime('now')
         WHERE id = ?`,
      ).bind(archive_key, draft_id),

      // Update chapter with prepared + mask prefixes and page count
      this.env.DB.prepare(
        `UPDATE chapters
         SET prepared_prefix = ?, masks_prefix = ?, page_count = ?, updated_at = datetime('now')
         WHERE id = ?`,
      ).bind(prepared_prefix, masks_prefix, page_count, chapter_id),
    ]);

    // Notify connected WebSocket clients
    await this._doNotify(draft_id, { type: "done", archive_key });
  }

  /**
   * Called by pipeline at each stage transition.
   * Stores progress in D1 (for late-loading UIs) and broadcasts via DO.
   */
  async notifyProgress(args: {
    draft_id:  number;
    stage:     PipelineStage;
    index?:    number;
    total?:    number;
  }): Promise<void> {
    const { draft_id, stage, index, total } = args;

    // Update D1 progress columns
    await this.env.DB.prepare(
      `UPDATE translation_drafts
       SET state = 'running', progress_stage = ?,
           progress_index = ?, progress_total = ?, updated_at = datetime('now')
       WHERE id = ?`,
    ).bind(stage, index ?? null, total ?? null, draft_id).run();

    await this._doNotify(draft_id, { type: "progress", stage, index, total });
  }

  /**
   * Called by pipeline when a stage fails unrecoverably.
   */
  async notifyError(args: {
    draft_id: number;
    stage:    PipelineStage;
    message:  string;
  }): Promise<void> {
    const { draft_id, stage, message } = args;

    await this.env.DB.prepare(
      `UPDATE translation_drafts
       SET state = 'error', error_message = ?, progress_stage = ?, updated_at = datetime('now')
       WHERE id = ?`,
    ).bind(message, stage, draft_id).run();

    await this._doNotify(draft_id, { type: "error", stage, message });
  }

  // ── Private ───────────────────────────────────────────────────────

  private async _doNotify(draft_id: number, event: PipelineProgressEvent): Promise<void> {
    const id   = this.env.STATUS_DO.idFromName(String(draft_id));
    const stub = this.env.STATUS_DO.get(id);
    // Fire and forget — don't block pipeline finalize on WebSocket delivery
    this.ctx.waitUntil(stub.notify(event));
  }
}
