/**
 * FanInCounterDO — atomic per-(workflow,stage) counter.
 *
 * SQLite-backed (works on Workers Free plan). Two operations:
 *   init(workflow_id, stage, n)  → set counter
 *   decrement(workflow_id, stage) → atomic decrement, returns true if hit 0
 *
 * Concurrent decrements from queue consumer isolates serialize on the DO.
 * SQL UPSERT in a single statement guarantees atomicity without
 * blockConcurrencyWhile (which would hurt throughput).
 *
 * Cleanup: each row gets an alarm 1h after init. If the workflow never
 * completes, the stale counter self-evicts.
 */

import { DurableObject } from "cloudflare:workers";

interface Env {}

export class FanInCounterDO extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // Schema init — only runs once per isolate; blockConcurrencyWhile is
    // appropriate here because no request handlers run until it returns.
    ctx.blockConcurrencyWhile(async () => {
      ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS counters (
          workflow_id TEXT NOT NULL,
          stage       TEXT NOT NULL,
          n           INTEGER NOT NULL,
          PRIMARY KEY (workflow_id, stage)
        )
      `);
    });
  }

  /** Set the counter for a fan-out. Called by the workflow before enqueuing. */
  async init(workflowId: string, stage: string, n: number): Promise<void> {
    this.ctx.storage.sql.exec(
      `INSERT INTO counters (workflow_id, stage, n) VALUES (?, ?, ?)
       ON CONFLICT (workflow_id, stage) DO UPDATE SET n = excluded.n`,
      workflowId, stage, n,
    );
    // Schedule cleanup 1h from now — stale counters self-evict if the
    // workflow dies before fan-in completes.
    await this.ctx.storage.setAlarm(Date.now() + 60 * 60 * 1000);
  }

  /**
   * Atomic decrement. Returns true if this call reduced the counter to 0
   * (the caller is the "last one home" and should notify).
   *
   * Single UPDATE statement → SQLite handles concurrency without explicit
   * transaction blocks. RETURNING gives us the new value in one round trip.
   */
  async decrement(workflowId: string, stage: string): Promise<boolean> {
    const rows = this.ctx.storage.sql.exec<{ n: number }>(
      `UPDATE counters SET n = n - 1
       WHERE workflow_id = ? AND stage = ? AND n > 0
       RETURNING n`,
      workflowId, stage,
    ).toArray();
    if (rows.length === 0) return false;     // already zero (shouldn't happen)
    const remaining = rows[0].n;
    if (remaining === 0) {
      // Last decrement — delete the row so a new fan-out can reuse the key.
      this.ctx.storage.sql.exec(
        `DELETE FROM counters WHERE workflow_id = ? AND stage = ?`,
        workflowId, stage,
      );
      return true;
    }
    return false;
  }

  /** Read current value — debugging only. */
  async peek(workflowId: string, stage: string): Promise<number> {
    const rows = this.ctx.storage.sql.exec<{ n: number }>(
      `SELECT n FROM counters WHERE workflow_id = ? AND stage = ?`,
      workflowId, stage,
    ).toArray();
    return rows.length > 0 ? rows[0].n : -1;
  }

  /** Alarm handler — drop expired counters. */
  override async alarm(): Promise<void> {
    // 1h cutoff: any counter older than that is orphaned. We don't track
    // age per row, but since each init() reschedules the alarm, an alarm
    // firing means at least 1h of inactivity → purge everything.
    this.ctx.storage.sql.exec(`DELETE FROM counters`);
  }
}
