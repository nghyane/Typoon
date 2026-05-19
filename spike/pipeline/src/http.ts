/** Public HTTP surface — chapter upload + workflow start + status polling.
 *
 * Three routes:
 *   PUT  /upload?key=raw/{chapter_id}/source.zip   → R2 streaming upload
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
      routes:  ["POST /start", "GET /status?id=…", "PUT /upload?key=…"],
    });
  }

  if (url.pathname === "/upload" && req.method === "PUT") {
    return uploadRaw(req, env);
  }

  if (url.pathname === "/download" && req.method === "GET") {
    return downloadR2(req, env);
  }

  if (url.pathname === "/start" && req.method === "POST") {
    return startWorkflow(req, env);
  }

  if (url.pathname === "/status" && req.method === "GET") {
    return statusWorkflow(req, env);
  }

  return new Response("not found", { status: 404 });
}

async function uploadRaw(req: Request, env: PipelineEnv): Promise<Response> {
  const key = new URL(req.url).searchParams.get("key");
  if (!key?.startsWith("raw/"))
    return new Response("key must start with raw/", { status: 400 });
  const ct = req.headers.get("Content-Type") ?? "application/octet-stream";
  await env.R2.put(key, req.body as ReadableStream, {
    httpMetadata: { contentType: ct },
  });
  return Response.json({ ok: true, key });
}

async function downloadR2(req: Request, env: PipelineEnv): Promise<Response> {
  const key = new URL(req.url).searchParams.get("key");
  if (!key) return new Response("missing key", { status: 400 });
  const obj = await env.R2.get(key);
  if (!obj) return new Response("not found", { status: 404 });
  return new Response(obj.body, {
    headers: {
      "Content-Type": obj.httpMetadata?.contentType ?? "application/octet-stream",
      "Content-Length": String(obj.size),
    },
  });
}

async function startWorkflow(req: Request, env: PipelineEnv): Promise<Response> {
  let params: PipelineParams;
  try { params = await req.json() as PipelineParams; }
  catch (e) { return Response.json({ error: String(e) }, { status: 400 }); }

  if (!params.chapter_id || !params.source_lang || !params.target_lang)
    return Response.json({ error: "chapter_id, source_lang, target_lang required" }, { status: 400 });
  if (!params.zip_key)
    return Response.json({ error: "zip_key required" }, { status: 400 });

  const instance = await env.PIPELINE.create({ params });
  return Response.json({ id: instance.id, status: await instance.status() });
}

async function statusWorkflow(req: Request, env: PipelineEnv): Promise<Response> {
  const id = new URL(req.url).searchParams.get("id");
  if (!id) return new Response("missing id", { status: 400 });
  const instance = await env.PIPELINE.get(id);
  return Response.json(await instance.status());
}
