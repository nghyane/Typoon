/**
 * Tiny R2 helpers. Workers pass an R2Bucket binding via `env.R2`.
 *
 * Keys follow the layout in docs/wiki/cloudflare-edge-pipeline.md:
 *   raw/{chapter}/{i:04d}.jpg
 *   prepared/{chapter}/{i:04d}.jpg
 *   prepared/{chapter}/meta.json
 *   scan/{chapter}/{i:04d}.json
 *   scan/{chapter}/{i:04d}.mask.png
 *   inpaint/{chapter}/{i:04d}.png
 *   translate/{chapter}.json
 *
 * Helpers below are tiny on purpose — workers compose them directly
 * with their own typed payloads. No DSL over R2.
 */

export interface R2Like {
  get(key: string): Promise<R2ObjectBody | null>;
  put(key: string, value: ArrayBuffer | ReadableStream | string | Uint8Array,
      options?: R2PutOptions): Promise<R2Object | null>;
  head(key: string): Promise<R2Object | null>;
  list(options?: R2ListOptions): Promise<R2Objects>;
}

export async function getBytes(r2: R2Like, key: string): Promise<Uint8Array> {
  const obj = await r2.get(key);
  if (!obj) throw new Error(`R2 miss: ${key}`);
  return new Uint8Array(await obj.arrayBuffer());
}

export async function getJson<T>(r2: R2Like, key: string): Promise<T> {
  const obj = await r2.get(key);
  if (!obj) throw new Error(`R2 miss: ${key}`);
  return await obj.json<T>();
}

export async function putBytes(r2: R2Like, key: string, bytes: Uint8Array, contentType: string): Promise<void> {
  await r2.put(key, bytes, { httpMetadata: { contentType } });
}

export async function putJson(r2: R2Like, key: string, value: unknown): Promise<void> {
  await r2.put(key, JSON.stringify(value), {
    httpMetadata: { contentType: "application/json" },
  });
}

export async function exists(r2: R2Like, key: string): Promise<boolean> {
  const head = await r2.head(key);
  return head !== null;
}

// Key builders.
export const keys = {
  raw:        (chap: string, i: number) => `raw/${chap}/${pad(i)}.jpg`,
  prepared:   (chap: string, i: number) => `prepared/${chap}/${pad(i)}.jpg`,
  preparedMeta: (chap: string)          => `prepared/${chap}/meta.json`,
  scan:       (chap: string, i: number) => `scan/${chap}/${pad(i)}.json`,
  scanMask:   (chap: string, i: number) => `scan/${chap}/${pad(i)}.mask.png`,
  inpaint:    (chap: string, i: number) => `inpaint/${chap}/${pad(i)}.png`,
  typeset:    (chap: string, i: number) => `typeset/${chap}/${pad(i)}.png`,
  translate:  (chap: string)            => `translate/${chap}.json`,
  // Brief outputs — split by invalidation frequency:
  //   prose       chapter-wide free-form, immutable per chapter
  //   glossary    source → target term map (mergeable cross-chapter)
  //   characters  per-character record (mergeable cross-chapter)
  //   address     speaker↔listener pronoun pairs (chapter-wide)
  //   keyNotes    per-bubble speaker hints (page-scoped)
  //   noise       per-bubble noise classification (translate + render)
  brief:        (chap: string)            => `brief/${chap}/index.json`,
  briefProse:   (chap: string)            => `brief/${chap}/prose.txt`,
  briefGloss:   (chap: string)            => `brief/${chap}/glossary.json`,
  briefChars:   (chap: string)            => `brief/${chap}/characters.json`,
  briefAddress: (chap: string)            => `brief/${chap}/address.json`,
  briefNotes:   (chap: string)            => `brief/${chap}/key_notes.json`,
  briefNoise:   (chap: string)            => `brief/${chap}/noise.json`,
  // Work-context I/O for the client → pipeline → client round-trip.
  // Input (optional): client gzip(JSON.stringify(WorkContext)) at job create.
  // Output: brief stage writes the merged WorkContext for client to fetch.
  ctxIn:        (chap: string)            => `ctx/${chap}/input.json.gz`,
  ctxOut:       (chap: string)            => `ctx/${chap}/output.json.gz`,
};

function pad(n: number): string { return String(n).padStart(4, "0"); }
