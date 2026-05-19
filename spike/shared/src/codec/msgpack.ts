/**
 * Minimal msgpack decoder for scan result payloads.
 *
 * Only handles the subset emitted by Python's msgpack.packb():
 *   - fixmap, map16, map32
 *   - fixarray, array16, array32
 *   - fixstr, str8, str16, str32
 *   - uint8..uint64, int8..int64, fixint, negfixint
 *   - nil, true, false
 *   - bin8, bin16, bin32 (returned as Uint8Array)
 *   - float32, float64
 *
 * Not supported: ext types (not used in scan payloads).
 * Throws on unknown format byte rather than silently skipping.
 */

export function decodeMsgpack<T = unknown>(buf: Uint8Array): T {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 0;

  function readByte(): number { return buf[pos++]; }
  function readU8(): number   { return buf[pos++]; }
  function readU16(): number  { const v = view.getUint16(pos); pos += 2; return v; }
  function readU32(): number  { const v = view.getUint32(pos); pos += 4; return v; }
  function readI8(): number   { const v = view.getInt8(pos); pos += 1; return v; }
  function readI16(): number  { const v = view.getInt16(pos); pos += 2; return v; }
  function readI32(): number  { const v = view.getInt32(pos); pos += 4; return v; }
  function readI64(): number  {
    const hi = view.getInt32(pos); const lo = view.getUint32(pos + 4);
    pos += 8; return hi * 0x100000000 + lo;
  }
  function readU64(): number  {
    const hi = view.getUint32(pos); const lo = view.getUint32(pos + 4);
    pos += 8; return hi * 0x100000000 + lo;
  }
  function readF32(): number  { const v = view.getFloat32(pos); pos += 4; return v; }
  function readF64(): number  { const v = view.getFloat64(pos); pos += 8; return v; }

  function readStr(len: number): string {
    const slice = buf.subarray(pos, pos + len); pos += len;
    return new TextDecoder().decode(slice);
  }
  function readBin(len: number): Uint8Array {
    const slice = buf.subarray(pos, pos + len); pos += len;
    return slice;
  }
  function readArray(len: number): unknown[] {
    const arr: unknown[] = new Array(len);
    for (let i = 0; i < len; i++) arr[i] = decode();
    return arr;
  }
  function readMap(len: number): Record<string, unknown> {
    const obj: Record<string, unknown> = {};
    for (let i = 0; i < len; i++) {
      const key = decode() as string;
      obj[key] = decode();
    }
    return obj;
  }

  function decode(): unknown {
    const byte = readByte();

    // positive fixint
    if ((byte & 0x80) === 0) return byte;
    // fixmap
    if ((byte & 0xf0) === 0x80) return readMap(byte & 0x0f);
    // fixarray
    if ((byte & 0xf0) === 0x90) return readArray(byte & 0x0f);
    // fixstr
    if ((byte & 0xe0) === 0xa0) return readStr(byte & 0x1f);
    // negative fixint
    if ((byte & 0xe0) === 0xe0) return byte - 256;

    switch (byte) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: return readBin(readU8());
      case 0xc5: return readBin(readU16());
      case 0xc6: return readBin(readU32());
      case 0xca: return readF32();
      case 0xcb: return readF64();
      case 0xcc: return readU8();
      case 0xcd: return readU16();
      case 0xce: return readU32();
      case 0xcf: return readU64();
      case 0xd0: return readI8();
      case 0xd1: return readI16();
      case 0xd2: return readI32();
      case 0xd3: return readI64();
      case 0xd9: return readStr(readU8());
      case 0xda: return readStr(readU16());
      case 0xdb: return readStr(readU32());
      case 0xdc: return readArray(readU16());
      case 0xdd: return readArray(readU32());
      case 0xde: return readMap(readU16());
      case 0xdf: return readMap(readU32());
      default:
        throw new Error(`msgpack: unknown byte 0x${byte.toString(16)} at pos ${pos - 1}`);
    }
  }

  return decode() as T;
}
