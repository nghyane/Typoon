/** Public HTTP surface — workflow start + status polling.
 *
 * Two routes:
 *   POST /start                                    → create workflow instance
 *   GET  /status?id=…                              → poll workflow status
 *
 * Everything else returns the service banner or 404. */

import type { PipelineEnv, PipelineParams } from "./pipeline";

export async function handleHttp(req: Request, env: PipelineEnv): Promise<Response> {
  const url = new URL(req.url);

  if (url.pathname === "/" && req.method === "GET") {
    return Response.json({
      ok:      true,
      service: "typoon-pipeline",
      routes:  ["POST /start", "GET /status?id=…"],
    });
  }

  if (url.pathname === "/start" && req.method === "POST") {
    return startWorkflow(req, env);
  }

  if (url.pathname === "/status" && req.method === "GET") {
    return statusWorkflow(req, env);
  }

  return new Response("not found", { status: 404 });
}

async function startWorkflow(req: Request, env: PipelineEnv): Promise<Response> {
  let body: any;
  try { body = await req.json(); }
  catch (e) { return Response.json({ error: String(e) }, { status: 400 }); }

  const job_id = Number(body.job_id);
  const kind   = body.kind === "analyze" ? "analyze" : "translate";
  const { source_lang, target_lang, zip_key, strategy, context_in_key } = body;

  if (isNaN(job_id) || !source_lang || !target_lang) {
    return Response.json({ error: "job_id (number), source_lang, target_lang required" }, { status: 400 });
  }
  if (!zip_key) {
    return Response.json({ error: "zip_key required" }, { status: 400 });
  }

  const params: PipelineParams = {
    job_id,
    kind,
    source_lang,
    target_lang,
    zip_key,
    strategy,
    context_in_key,
  };

  const instance = await env.PIPELINE.create({ params });
  return Response.json({ id: instance.id, status: await instance.status() });
}

async function statusWorkflow(req: Request, env: PipelineEnv): Promise<Response> {
  const id = new URL(req.url).searchParams.get("id");
  if (!id) return new Response("missing id", { status: 400 });
  const instance = await env.PIPELINE.get(id);
  return Response.json(await instance.status());
}
