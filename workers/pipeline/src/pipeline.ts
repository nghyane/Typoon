/** ChapterPipeline — simplified workflow.
 *
 *   prepare         → MediaService.prepareChapter
 *   scan-chapter    → ScanService.scanChapter (container: comic_detr + Lens + spatial_join)
 *                     + storyboard (concurrent, same container)
 *                     + enqueue inpaints immediately
 *   brief           → BriefService (reads storyboard + scan msgpacks)
 *   translate       → concurrent with inpaints
 *   waitForEvent("inpaints-done")
 *   typeset-pack    → TypesetPackService (render + BNL in one container)
 */

import {
  WorkflowEntrypoint, type WorkflowStep, type WorkflowEvent,
} from "cloudflare:workers";

import type { FanInCounterDO } from "./fan-in";
import { K, type InpaintsDonePayload } from "./types";
import type {
  MediaService, ScanService, BriefService, TranslateService,
  TypesetPackService, ApiCallbackService, Service
} from "./services";

interface InpaintJob {
  workflow_id:  string;
  chapter_id:   number;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  scan_key:     string;
  mask_key:     string;
}

export interface PipelineEnv {
  MEDIA:         Service<MediaService>;
  SCAN:          Service<ScanService>;
  BRIEF:         Service<BriefService>;
  TRANSLATE:     Service<TranslateService>;
  TYPESET_PACK:  Service<TypesetPackService>;
  API:           Service<ApiCallbackService>;
  INPAINT_QUEUE: Queue<InpaintJob>;
  FAN_IN:        DurableObjectNamespace<FanInCounterDO>;
  R2:            R2Bucket;
  PIPELINE:      Workflow;
}

export interface PipelineParams {
  chapter_id:  number;
  draft_id:    number;
  source_lang: string;
  target_lang: string;
  zip_key:     string;
  strategy?:   "auto" | "one_to_one" | "stitch";
}

export class ChapterPipeline extends WorkflowEntrypoint<PipelineEnv, PipelineParams> {
  declare env: PipelineEnv;

  override async run(event: WorkflowEvent<PipelineParams>, step: WorkflowStep) {
    const { chapter_id, draft_id, source_lang, target_lang, zip_key, strategy } = event.payload;
    const workflowId = event.instanceId;

    try {
      // ── 1. Prepare ────────────────────────────────────────────────────────────
      const prepared = await step.do("prepare", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "prepare" });
        const result = await this.env.MEDIA.prepareChapter({ chapter_id, zip_key, strategy });
        // Init fan-in counter here to avoid a separate step round-trip (~1s overhead).
        const fanIn = this.env.FAN_IN.getByName(workflowId);
        await fanIn.init(workflowId, "inpaints-done", result.pages.length);
        return result;
      });
      const N = prepared.pages.length;
      const fanIn = this.env.FAN_IN.getByName(workflowId);

      // ── 2. Scan Chapter ───────────────────────────────────────────────────────
      const scanResult = await step.do("scan-chapter", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "scan" });
        const result = await this.env.SCAN.scanChapter({
          chapter_id,
          workflow_id: workflowId,
          pages: prepared.pages.map(p => ({
            page_index:   p.index,
            prepared_key: K.prepared(chapter_id, p.index),
            is_color:     prepared.is_color,
          })),
          lang_hint:   source_lang,
          total_pages: N,
        });

        // Enqueue inpaints immediately
        const jobs = result.scan_keys.map((scan_key, i) => ({
          workflow_id: workflowId,
          chapter_id,
          page_index:  prepared.pages[i]!.index,
          total_pages: N,
          prepared_key: K.prepared(chapter_id, prepared.pages[i]!.index),
          scan_key,
          mask_key: result.mask_keys[i] || "",
        }));

        for (let i = 0; i < jobs.length; i += 100) {
          await this.env.INPAINT_QUEUE.sendBatch(
            jobs.slice(i, i + 100).map(body => ({ body }))
          );
        }
        return result;
      });

      const { scan_keys, mask_keys, storyboard_keys } = scanResult;

      // ── 3. Brief ──────────────────────────────────────────────────────────────
      const brief = await step.do("brief", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "brief" });
        return await this.env.BRIEF.briefChapter({
          chapter_id,
          source_lang,
          target_lang,
          is_color: prepared.is_color,
          strategy: prepared.strategy,
          scan_keys,
          storyboard_keys,
        });
      });

      // ── 4. Translate ──────────────────────────────────────────────────────────
      const translate = await step.do("translate", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "translate" });
        return await this.env.TRANSLATE.translateChapter({
          chapter_id,
          scan_keys,
          source_lang,
          target_lang,
          use_brief: true,
        });
      });

      // ── 5. Wait for inpaints ──────────────────────────────────────────────────
      const inpaintsEvent = await step.waitForEvent<InpaintsDonePayload>("await-inpaints", {
        type: "inpaints-done",
        timeout: "15 minutes",
      });
      const { inpaint_keys } = inpaintsEvent.payload;

      // ── 6. Typeset + Pack ─────────────────────────────────────────────────────
      const archive = await step.do("typeset-pack", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "typeset" });
        return await this.env.TYPESET_PACK.typesetAndPack({
          chapter_id,
          translate_key: translate.output_key,
          pages: prepared.pages.map((p, i) => ({
            page_index:  p.index,
            inpaint_key: inpaint_keys[i] || "",
            scan_key:    scan_keys[i] || "",
            page_width:  p.width,
          })),
        });
      });

      // ── 7. Finalize ───────────────────────────────────────────────────────────
      await step.do("finalize", async () => {
        await this.env.API.notifyProgress({ draft_id, stage: "finalize" });
        await this.env.API.finalize({
          chapter_id,
          draft_id,
          archive_key: archive.archive_key,
          page_count:  N,
          scan_keys,
          mask_keys,
        });
      });

      return {
        chapter_id,
        pages: N,
        is_color: prepared.is_color,
        strategy: prepared.strategy,
        scan_keys,
        mask_keys,
        inpaint_keys,
        storyboard_keys,
        brief_index_key: brief.index_key,
        translate_key:   translate.output_key,
        archive_key:     archive.archive_key,
        archive_size:    archive.size_bytes,
      };
    } catch (err: any) {
      // Notify API about error
      try {
        await this.env.API.notifyError({
          draft_id,
          stage: "finalize", // default to last stage or extract from current step
          message: err.message || String(err),
        });
      } catch (notifyErr) {
        console.error("Failed to notify error", notifyErr);
      }
      throw err; // rethrow to fail the workflow run
    }
  }
}
