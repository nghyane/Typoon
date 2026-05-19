#!/usr/bin/env node
/**
 * e2e.mjs — End-to-end test for typoon-pipeline.
 *
 * 1. Fetch a chapter from MangaDex (first JP chapter of a popular series)
 * 2. Download all page images, pack into a zip
 * 3. Upload zip to R2 via PUT /upload
 * 4. POST /start  → workflow instance
 * 5. Poll /status until done or timeout (20 min)
 *
 * Usage:
 *   node e2e.mjs [chapter_id]
 *   node e2e.mjs  # picks a default test chapter
 */

import { createWriteStream, readFileSync } from "fs";
import { writeFile, mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { randomUUID } from "crypto";

// ── Config ────────────────────────────────────────────────────────────────────

const PIPELINE_URL = "https://typoon-pipeline.hoangvananhnghia99.workers.dev";
// Popular JP manga chapter on MangaDex for testing — Chainsaw Man ch.1 (JP)
const DEFAULT_CHAPTER_ID = "6fd9bac8-3003-4e0b-982e-40ded61b4194"; // 28-page JP chapter
const SOURCE_LANG = "ja";
const TARGET_LANG = "vi";

// ── Helpers ───────────────────────────────────────────────────────────────────

async function fetchJson(url, opts = {}, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const r = await fetch(url, opts);
      if (!r.ok) throw new Error(`${opts.method ?? "GET"} ${url} → ${r.status}: ${await r.text()}`);
      return r.json();
    } catch (e) {
      if (i === retries - 1) throw e;
      console.warn(`\n[retry ${i+1}] ${e.message}`);
      await sleep(3000);
    }
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function fmt(obj) { return JSON.stringify(obj, null, 2); }

// ── MangaDex helpers ──────────────────────────────────────────────────────────

async function resolveChapter(chapterId) {
  console.log(`[mangadex] fetching chapter metadata: ${chapterId}`);
  const data = await fetchJson(`https://api.mangadex.org/chapter/${chapterId}`);
  const ch   = data.data;
  const title = ch.attributes.title || `ch${ch.attributes.chapter}`;
  console.log(`[mangadex] chapter: ${title} (${ch.attributes.translatedLanguage})`);
  return { id: ch.id, title };
}

async function getPageUrls(chapterId) {
  console.log(`[mangadex] fetching at-home server for ${chapterId}`);
  const data = await fetchJson(`https://api.mangadex.org/at-home/server/${chapterId}`);
  const base = data.baseUrl;
  const hash = data.chapter.hash;
  const pages = data.chapter.data; // high-res pages
  return pages.map(p => `${base}/data/${hash}/${p}`);
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const chapterId = process.argv[2] || DEFAULT_CHAPTER_ID;

  // 1. Resolve + pages
  await resolveChapter(chapterId);
  const pageUrls = await getPageUrls(chapterId);
  console.log(`[mangadex] ${pageUrls.length} pages`);

  // 2. Download pages + build zip in memory using JSZip-compatible approach
  //    We'll use Node's built-in zip via the 'archiver' approach —
  //    but keep deps-free: write raw images to tmp dir then zip with `zip` CLI.
  const tmpDir = join(tmpdir(), `typoon-e2e-${randomUUID()}`);
  await mkdir(tmpDir, { recursive: true });

  console.log(`[download] downloading ${pageUrls.length} pages to ${tmpDir} ...`);
  await Promise.all(pageUrls.map(async (url, i) => {
    const ext = url.split(".").pop().split("?")[0] || "jpg";
    const dest = join(tmpDir, `${String(i).padStart(4, "0")}.${ext}`);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`page ${i} fetch failed: ${r.status}`);
    await writeFile(dest, Buffer.from(await r.arrayBuffer()));
    process.stdout.write(".");
  }));
  console.log("\n[download] done");

  // 3. Zip
  const zipPath = join(tmpdir(), `typoon-e2e-${chapterId}.zip`);
  const { execSync } = await import("child_process");
  execSync(`cd "${tmpDir}" && zip -r "${zipPath}" .`, { stdio: "inherit" });
  console.log(`[zip] created ${zipPath}`);

  // 4. Upload to pipeline R2
  const zipKey  = `raw/${chapterId}/source.zip`;
  const zipData = readFileSync(zipPath);
  console.log(`[upload] uploading ${(zipData.length / 1024 / 1024).toFixed(1)} MB → ${zipKey}`);
  await fetchJson(`${PIPELINE_URL}/upload?key=${encodeURIComponent(zipKey)}`, {
    method:  "PUT",
    headers: { "Content-Type": "application/zip" },
    body:    zipData,
  });
  console.log("[upload] done");

  // 5. Start workflow
  const params = {
    chapter_id:  chapterId,
    source_lang: SOURCE_LANG,
    target_lang: TARGET_LANG,
    zip_key:     zipKey,
  };
  console.log("[pipeline] starting workflow ...");
  const started = await fetchJson(`${PIPELINE_URL}/start`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(params),
  });
  console.log("[pipeline] started:", fmt(started));
  const instanceId = started.id;

  // 6. Poll until complete / failed / timeout
  const deadline = Date.now() + 20 * 60 * 1000;
  let last = "";
  while (Date.now() < deadline) {
    await sleep(5000);
    const status = await fetchJson(`${PIPELINE_URL}/status?id=${instanceId}`);
    const state  = status.status;
    if (state !== last) {
      console.log(`[status] ${state}`, status.output ? fmt(status.output) : "");
      last = state;
    } else {
      process.stdout.write(".");
    }
    if (state === "complete") {
      console.log("\n[e2e] ✓ DONE");
      console.log(fmt(status.output));
      break;
    }
    if (state === "errored" || state === "failed" || state === "terminated") {
      console.error("\n[e2e] FAILED:", fmt(status));
      process.exit(1);
    }
  }
  if (Date.now() >= deadline) {
    console.error("[e2e] TIMEOUT after 20 min");
    process.exit(1);
  }

  // Cleanup
  await rm(tmpDir, { recursive: true, force: true });
}

main().catch(e => { console.error(e); process.exit(1); });
