/**
 * PipelineNotifier — WorkerEntrypoint exposed to queue consumer workers.
 *
 * Queue consumers (scan, inpaint) call notify() after their FanInCounterDO
 * decrement returns true. notify() sends an event to the workflow instance,
 * waking it from step.waitForEvent().
 *
 * Why a separate entrypoint instead of consumers calling the Workflow
 * binding directly?
 *   - Workflow binding (env.PIPELINE) lives in this worker — consumers
 *     in other workers cannot bind it directly.
 *   - Service binding → WorkerEntrypoint → Workflow.get().sendEvent()
 *     is the clean cross-worker pattern.
 */

import { WorkerEntrypoint } from "cloudflare:workers";

interface Env {
  PIPELINE: Workflow;
}

export class PipelineNotifier extends WorkerEntrypoint<Env> {
  async notify(workflowId: string, eventType: string, payload: unknown): Promise<void> {
    const instance = await this.env.PIPELINE.get(workflowId);
    await instance.sendEvent({ type: eventType, payload });
  }
}
