/**
 * PipelineCallback — WorkerEntrypoint RPC called by typoon-pipeline.
 *
 * Pipeline calls:
 *   env.API.notifyProgress({ job_id, stage, ... })
 *   env.API.finalize({ job_id, archive_key, page_count })
 *   env.API.notifyError({ job_id, stage, message })
 *
 * After each D1 mutation we ask the per-user DO to push the fresh `ApiJob`
 * to every device the user has open. One WS per user multiplexes all jobs.
 */

import { WorkerEntrypoint } from "cloudflare:workers";
import type { Env, FinalizeArgs, PipelineStage } from "../types";
import { setJobState, recordConsume, getJob } from "../store/jobs";
import { copyContextFromR2ToKv } from "../store/work-context";

export class PipelineCallback extends WorkerEntrypoint<Env> {
  async finalize(args: FinalizeArgs): Promise<void> {
    const { job_id, archive_key, page_count, context_out_key } = args;

    const job = await getJob(this.env.DB, job_id);
    if (!job) return;

    await setJobState(this.env.DB, job_id, "done", {
      archive_key,
      page_count,
      context_out_key,
      progress_stage: "finalize",
      finished_at:    true,
    });

    // Bill the quota — only reached when prepare succeeded.
    await recordConsume(this.env.DB, {
      user_id:    job.user_id,
      job_id,
      page_count,
    });

    // Auto-merge: pipe brief output → KV ctx:{user}:{work_id}.
    // This is what makes contexts share across Web + Ext: any client that
    // submits a job with the same work_id will pull the latest version on
    // the next POST /jobs.
    if (job.work_id && context_out_key) {
      try {
        await copyContextFromR2ToKv(
          this.env, job.user_id, job.work_id, context_out_key,
        );
      } catch (err) {
        console.error("finalize: KV merge failed", { job_id, work_id: job.work_id, err });
        // Job still 'done' — context just didn't propagate.
        // Client can re-fetch context_out_url from R2 to recover.
      }
    }

    await this._publish(job.user_id, job_id);
  }

  async notifyProgress(args: {
    job_id: number;
    stage:  PipelineStage;
    index?: number;
    total?: number;
  }): Promise<void> {
    await setJobState(this.env.DB, args.job_id, "running", {
      progress_stage: args.stage,
      progress_index: args.index,
      progress_total: args.total,
    });
    const job = await getJob(this.env.DB, args.job_id);
    if (job) await this._publish(job.user_id, args.job_id);
  }

  async notifyError(args: {
    job_id:  number;
    stage:   PipelineStage;
    message: string;
  }): Promise<void> {
    await setJobState(this.env.DB, args.job_id, "error", {
      error_message:  args.message,
      progress_stage: args.stage,
      finished_at:    true,
    });

    // Do NOT charge quota on errors. The job row keeps the failure
    // for the user to inspect; chapter_consumes is not written.

    const job = await getJob(this.env.DB, args.job_id);
    if (job) await this._publish(job.user_id, args.job_id);
  }

  /** Ask the per-user DO to broadcast a fresh `ApiJob` to every session.
   *  Best-effort: the next request from the client will pull the same row
   *  from D1 anyway, so a DO error must never propagate. */
  private async _publish(user_id: number, job_id: number): Promise<void> {
    const id   = this.env.USER_EVENTS_DO.idFromName(String(user_id));
    const stub = this.env.USER_EVENTS_DO.get(id);
    this.ctx.waitUntil(stub.publishJob(job_id));
  }
}
