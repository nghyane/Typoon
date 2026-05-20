/** typoon-pipeline — entry point.
 *
 *   default fetch()           → HTTP control surface (start/status)
 *   class ChapterPipeline     → WorkflowEntrypoint (binding "PIPELINE")
 *   class FanInCounterDO      → DurableObject (binding "FAN_IN")
 *   class PipelineNotifier    → WorkerEntrypoint exposed to scan/inpaint
 *                                queue consumers via service binding.
 */

export { FanInCounterDO } from "./fan-in";
export { PipelineNotifier } from "./notifier";
export { ChapterPipeline, type PipelineEnv } from "./pipeline";

import { handleHttp } from "./http";
import type { PipelineEnv } from "./pipeline";

export default {
  fetch: (req: Request, env: PipelineEnv): Promise<Response> => handleHttp(req, env),
};
