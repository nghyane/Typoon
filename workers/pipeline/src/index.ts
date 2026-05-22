/** typoon-pipeline — entry point.
 *
 *   default fetch()        → HTTP control surface (start/status)
 *   class ChapterPipeline  → WorkflowEntrypoint (binding "PIPELINE")
 *
 * No DOs, no queue producer, no custom RPC entrypoints beyond the workflow
 * itself. Coordination across pages is `Promise.all` over `step.do` inside
 * `ChapterPipeline.run` — see pipeline.ts.
 */

export { ChapterPipeline, type PipelineEnv } from "./pipeline";

import { handleHttp } from "./http";
import type { PipelineEnv } from "./pipeline";

export default {
  fetch: (req: Request, env: PipelineEnv): Promise<Response> => handleHttp(req, env),
};
