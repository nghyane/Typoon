/**
 * Opaque translation key generation — port of typoon/stages/keys.py.
 *
 * Key is the single identity for LLM communication. Stable across runs
 * (same chapter_id + page + idx → same key) but unguessable.
 *
 *   payload = JSON({chapter_id, page, idx, salt}, sort_keys=true)
 *   raw     = utf8(payload)
 *   n       = blake2s(raw, 5 bytes).int_be
 *   key     = base32-of-(ABCDEFGHJKLMNPQRSTUVWXYZ23456789) — 7 chars
 *
 * Collision: bump `salt` until unique within the chapter.
 *
 * Web Crypto does not expose BLAKE2s. We bundle a small pure-JS impl
 * (`blake2s` below, 30 lines, BSD-licensed reference). This is a hot
 * path called once per bubble per request — JS speed is fine for the
 * tens-to-hundreds of bubbles a chapter ships.
 */

const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

export interface BubbleKeyArgs {
  chapter_id: number | string;
  page_index: number;
  idx:        number;
}

export function assignKey(args: BubbleKeyArgs, used: Set<string>): string {
  let salt = 0;
  while (true) {
    const k = makeKey(args, salt);
    if (!used.has(k)) {
      used.add(k);
      return k;
    }
    salt++;
  }
}

function makeKey(args: BubbleKeyArgs, salt: number): string {
  // Match Python: json.dumps(payload, sort_keys=True, ensure_ascii=False)
  // ⇒ keys alphabetically: chapter_id, idx, page, salt
  const payload = `{"chapter_id": ${jsonNum(args.chapter_id)}, "idx": ${args.idx}, "page": ${args.page_index}, "salt": ${salt}}`;
  const raw = new TextEncoder().encode(payload);
  const digest = blake2s(raw, 5);  // 5 bytes
  let n = 0n;
  for (const b of digest) n = (n << 8n) | BigInt(b);
  const out: string[] = [];
  const base = BigInt(ALPHABET.length);
  for (let i = 0; i < 7; i++) {
    const r = n % base;
    n = n / base;
    out.push(ALPHABET[Number(r)]);
  }
  return out.join("");
}

function jsonNum(v: number | string): string {
  if (typeof v === "number") return String(v);
  // Python json.dumps wraps strings in quotes; chapter_id is an int in the
  // Python pipeline, but tolerate string IDs here for the worker spike.
  return JSON.stringify(v);
}

// ─────────────────────────────────────────────────────────────────────────────
// BLAKE2s — RFC 7693, no-key variant. Output length 1..32 bytes.
//
// Adapted from the reference pseudocode. ~50 lines. Single-call API.
// ─────────────────────────────────────────────────────────────────────────────

const IV = new Uint32Array([
  0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
  0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]);

const SIGMA = [
  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
  [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
  [11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4],
  [7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8],
  [9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13],
  [2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9],
  [12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11],
  [13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10],
  [6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5],
  [10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0],
];

function rotr32(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

function G(v: Uint32Array, a: number, b: number, c: number, d: number, x: number, y: number): void {
  v[a] = (v[a] + v[b] + x) >>> 0;
  v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = (v[c] + v[d]) >>> 0;
  v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = (v[a] + v[b] + y) >>> 0;
  v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = (v[c] + v[d]) >>> 0;
  v[b] = rotr32(v[b] ^ v[c], 7);
}

function compress(h: Uint32Array, block: Uint32Array, t: number, last: boolean): void {
  const v = new Uint32Array(16);
  v.set(h, 0);
  v.set(IV, 8);
  v[12] ^= (t & 0xffffffff) >>> 0;
  v[13] ^= Math.floor(t / 0x100000000) >>> 0;
  if (last) v[14] = ~v[14] >>> 0;
  for (let i = 0; i < 10; i++) {
    const s = SIGMA[i];
    G(v, 0, 4,  8, 12, block[s[0]],  block[s[1]]);
    G(v, 1, 5,  9, 13, block[s[2]],  block[s[3]]);
    G(v, 2, 6, 10, 14, block[s[4]],  block[s[5]]);
    G(v, 3, 7, 11, 15, block[s[6]],  block[s[7]]);
    G(v, 0, 5, 10, 15, block[s[8]],  block[s[9]]);
    G(v, 1, 6, 11, 12, block[s[10]], block[s[11]]);
    G(v, 2, 7,  8, 13, block[s[12]], block[s[13]]);
    G(v, 3, 4,  9, 14, block[s[14]], block[s[15]]);
  }
  for (let i = 0; i < 8; i++) h[i] = (h[i] ^ v[i] ^ v[i + 8]) >>> 0;
}

export function blake2s(input: Uint8Array, outLen = 32): Uint8Array {
  if (outLen < 1 || outLen > 32) throw new Error("blake2s outLen 1..32");
  const h = new Uint32Array(IV);
  h[0] ^= 0x01010000 ^ outLen;

  const block = new Uint8Array(64);
  const view  = new DataView(block.buffer);
  const words = new Uint32Array(16);

  let off = 0, t = 0;
  const len = input.length;
  // Full blocks except last
  while (len - off > 64) {
    block.set(input.subarray(off, off + 64));
    t += 64;
    for (let i = 0; i < 16; i++) words[i] = view.getUint32(i * 4, true);
    compress(h, words, t, false);
    off += 64;
  }
  // Final block (always; even if input is empty)
  block.fill(0);
  const tail = input.subarray(off);
  block.set(tail);
  t += tail.length;
  for (let i = 0; i < 16; i++) words[i] = view.getUint32(i * 4, true);
  compress(h, words, t, true);

  const out = new Uint8Array(outLen);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < outLen; i++) {
    out[i] = (h[i >> 2] >>> ((i & 3) * 8)) & 0xff;
  }
  return out;
}
