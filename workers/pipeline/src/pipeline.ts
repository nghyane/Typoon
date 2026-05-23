/** ChapterPipeline — one workflow, one code path.
 *
 *   prepare         → MediaService.prepareChapter
 *   scan            → ScanService.scanChapter (detector + Lens + storyboard)
 *   [brief, translate] in parallel:
 *     brief         → BriefService (storyboard + scan msgpacks → noise/glossary/…)
 *     translate     → TranslateService (scan msgpacks → translate/{job}.json)
 *   if kind=analyze → finalize-analyze, return
 *   inpaint-NNNN    → InpaintService.inpaintPage, N parallel step.do (one per page).
 *                     Per-step retry/timeout config replaces the old queue+DLQ+
 *                     fan-in DO+event hand-off — no manual coordination needed.
 *   typeset-pack    → TypesetPackService.typesetAndPack
 *   finalize        → ApiCallbackService.finalize (writes done state + KV merge)
 *
 * Failure flow: any step.do throwing after its retry budget marks the step
 * `failed` in the workflow dashboard with the real error message, and the
 * top-level catch reports `currentStage` to ApiCallbackService.notifyError.
 * No silent paths, no global timeouts beyond per-step.
 */

import {
  WorkflowEntrypoint, type WorkflowStep, type WorkflowEvent,
  type WorkflowStepConfig,
} from "cloudflare:workers";

import { K } from "./types";
import type {
  MediaService, ScanService, BriefService, TranslateService,
  InpaintService, TypesetPackService, ApiCallbackService, Service,
} from "./services";

export interface PipelineEnv {
  MEDIA:        Service<MediaService>;
  SCAN:         Service<ScanService>;
  BRIEF:        Service<BriefService>;
  TRANSLATE:    Service<TranslateService>;
  INPAINT:      Service<InpaintService>;
  TYPESET_PACK: Service<TypesetPackService>;
  API:          Service<ApiCallbackService>;
  R2:           R2Bucket;
  PIPELINE:     Workflow;
}

export interface PipelineParams {
  job_id:          number;
  /** 'translate' runs the full pipeline; 'analyze' stops after brief
   *  and returns the merged context only (no archive). */
  kind:            "translate" | "analyze";
  source_lang:     string;
  target_lang:     string;
  zip_key:         string;
  /** Optional R2 key of client-supplied WorkContext (gzip+JSON). */
  context_in_key?: string;
  strategy?:       "auto" | "one_to_one" | "stitch";
}

// ── Per-stage step config ─────────────────────────────────────────────────────
// `retries.delay` ≥ 1s and `timeout` is the per-attempt budget. Numbers tuned to
// observed p95 latencies (prepare 16s, scan 50s, brief 38s, translate 42s).

const STEP: Record<string, WorkflowStepConfig> = {
  prepare:     { retries: { limit: 2, delay: "5 seconds", backoff: "exponential" }, timeout: "5 minutes" },
  scan:        { retries: { limit: 1, delay: "10 seconds" },                        timeout: "10 minutes" },
  brief:       { retries: { limit: 1, delay: "10 seconds" },                        timeout: "5 minutes" },
  translate:   { retries: { limit: 1, delay: "10 seconds" },                        timeout: "10 minutes" },
  inpaint:     { retries: { limit: 2, delay: "5 seconds", backoff: "exponential" }, timeout: "10 minutes" },
  typesetPack: { retries: { limit: 1, delay: "10 seconds" },                        timeout: "10 minutes" },
  finalize:    { retries: { limit: 3, delay: "2 seconds" },                         timeout: "1 minute" },
};

export class ChapterPipeline extends WorkflowEntrypoint<PipelineEnv, PipelineParams> {
  declare env: PipelineEnv;

  override async run(event: WorkflowEvent<PipelineParams>, step: WorkflowStep) {
    const { job_id, kind, source_lang, target_lang, zip_key, strategy, context_in_key } = event.payload;

    // `currentStage` is what we report to API.notifyError on top-level catch.
    // Mutated before each step.do so the user sees the real failing stage.
    let currentStage = "prepare";

    try {
      // ── 1. Prepare ─────────────────────────────────────────────────────────
      const prepared = await step.do("prepare", STEP.prepare, async () => {
        await this.env.API.notifyProgress({ job_id, stage: "prepare" });
        return this.env.MEDIA.prepareChapter({ job_id, zip_key, strategy });
      });
      const N = prepared.pages.length;

      // ── 2. Scan ───────────────────────────────────────────────────────────
      currentStage = "scan";
      const scan = await step.do("scan", STEP.scan, async () => {
        await this.env.API.notifyProgress({ job_id, stage: "scan", total: N });
        return this.env.SCAN.scanChapter({
          job_id,
          pages: prepared.pages.map(p => ({
            page_index:   p.index,
            prepared_key: K.prepared(job_id, p.index),
            is_color:     prepared.is_color,
          })),
          lang_hint:   source_lang,
          total_pages: N,
        });
      });

      // ── 3. Brief + Translate + Inpaint in parallel ─────────────────────────
      // Brief reads scan + storyboard keys; translate reads scan msgpacks
      // (+ optional noise from brief output, lazily); inpaint reads
      // prepared + mask + scan. None of the three depend on each other's
      // outputs — they only depend on the scan stage being done. We fire
      // all three together so the chapter wall time = max(brief, translate,
      // inpaint) instead of brief+translate then inpaint sequentially.
      //
      // Inpaint is sharded across container instances inside the same
      // Promise.all: each shard is its own checkpointed step.do so a
      // single page failure retries only that shard, not the whole
      // chapter, and brief/translate aren't affected.
      currentStage = "scan-done";
      await this.env.API.notifyProgress({ job_id, stage: "brief" });

      const SHARD_COUNT = Math.min(prepared.pages.length, 6);
      const shards: number[][] = Array.from({ length: SHARD_COUNT }, () => []);
      prepared.pages.forEach((p, i) => shards[i % SHARD_COUNT]!.push(p.index));

      const briefPromise = step.do("brief", STEP.brief, () => this.env.BRIEF.briefJob({
        job_id,
        source_lang,
        target_lang,
        is_color:        prepared.is_color,
        strategy:        prepared.strategy,
        scan_keys:       scan.scan_keys,
        storyboard_keys: scan.storyboard_keys,
        context_in_key,
      }));

      const translatePromise = kind === "translate"
        ? step.do("translate", STEP.translate, () => this.env.TRANSLATE.translateChapter({
            job_id,
            scan_keys:   scan.scan_keys,
            source_lang,
            target_lang,
            use_brief:   true,
          }))
        : Promise.resolve(null);

      // Inpaint is full-pipeline only — analyze skips render entirely.
      const inpaintShardsPromise: Promise<
        { results: { page_index: number; output_key?: string; error?: string }[] }[]
      > = kind === "translate"
        ? Promise.all(shards.map((indices, shardIdx) =>
            step.do(
              `inpaint-shard-${shardIdx}`,
              STEP.inpaint,
              () => this.env.INPAINT.inpaintChapter({
                job_id,
                page_indices: indices,
                concurrency:  4,
              }),
            ),
          ))
        : Promise.resolve([]);

      const [brief, translate, shardOutcomes] = await Promise.all([
        briefPromise,
        translatePromise,
        inpaintShardsPromise,
      ]);

      // ── Analyze short-circuit ──────────────────────────────────────────────
      if (kind === "analyze") {
        currentStage = "finalize";
        await step.do("finalize", STEP.finalize, async () => {
          await this.env.API.notifyProgress({ job_id, stage: "finalize" });
          await this.env.API.finalize({
            job_id,
            page_count:      N,
            context_out_key: brief.context_out_key,
          });
        });
        return {
          job_id,
          kind:            "analyze" as const,
          pages:           N,
          is_color:        prepared.is_color,
          strategy:        prepared.strategy,
          context_out_key: brief.context_out_key,
        };
      }

      // Verify every inpaint page succeeded. A single error fails the
      // whole pipeline — pipeline.catch() then calls API.notifyError with
      // the real reason.
      const byPage = new Map<number, string>();
      const failures: string[] = [];
      for (const shard of shardOutcomes) {
        for (const r of shard.results) {
          if (r.error || !r.output_key) {
            failures.push(`page ${r.page_index}: ${r.error ?? "missing output_key"}`);
          } else {
            byPage.set(r.page_index, r.output_key);
          }
        }
      }
      if (failures.length > 0) {
        currentStage = "inpaint";
        throw new Error(`inpaint failures (${failures.length}/${N}): ${failures.slice(0, 5).join("; ")}`);
      }
      const inpaint_keys = prepared.pages.map(p => byPage.get(p.index) ?? "");

      // ── 5. Typeset + Pack ──────────────────────────────────────────────────
      currentStage = "typeset";
      const archive = await step.do("typeset-pack", STEP.typesetPack, async () => {
        await this.env.API.notifyProgress({ job_id, stage: "typeset" });
        return this.env.TYPESET_PACK.typesetAndPack({
          job_id,
          translate_key: translate!.output_key,
          pages: prepared.pages.map((p, i) => ({
            page_index:  p.index,
            inpaint_key: inpaint_keys[i] || "",
            scan_key:    scan.scan_keys[i] || "",
            page_width:  p.width,
          })),
        });
      });

      // ── 6. Finalize ────────────────────────────────────────────────────────
      currentStage = "finalize";
      await step.do("finalize", STEP.finalize, async () => {
        await this.env.API.notifyProgress({ job_id, stage: "finalize" });
        await this.env.API.finalize({
          job_id,
          archive_key:     archive.archive_key,
          page_count:      N,
          context_out_key: brief.context_out_key,
        });
      });

      return {
        job_id,
        pages: N,
        is_color: prepared.is_color,
        strategy: prepared.strategy,
        scan_keys: scan.scan_keys,
        inpaint_keys,
        storyboard_keys: scan.storyboard_keys,
        brief_index_key: brief.index_key,
        translate_key:   translate!.output_key,
        archive_key:     archive.archive_key,
        archive_size:    archive.size_bytes,
      };
    } catch (err) {
      // The workflow runtime already marked the failing step `failed` with the
      // real error text. We mirror that to the API for the user-facing job
      // record. Best-effort: a notifyError failure must not mask the original.
      const message = err instanceof Error ? err.message : String(err);
      try {
        await this.env.API.notifyError({ job_id, stage: currentStage, message });
      } catch (notifyErr) {
        console.error("notifyError failed; writing R2 fallback marker", notifyErr);
        try {
          await this.env.R2.put(`jobs/${job_id}/error.json`, JSON.stringify({
            error_message: message,
            stage:         currentStage,
            timestamp:     new Date().toISOString(),
          }));
        } catch (r2Err) {
          console.error("R2 error marker write failed", r2Err);
        }
      }
      throw err;
    }
  }
}
