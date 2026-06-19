import type { TranslatedUnit } from '../../domain/translation'
import type { Translator } from '../translator'
import { AsyncLimiter } from '../../flow/AsyncLimiter'
import { batchUnits, parseTranslatedBatch, serializeBatch, toTranslatedUnit } from '../../pipeline/translation/MarkerProtocol'

export interface DeepLTranslateWebOptions {
  readonly endpoint?: string
  readonly requestTimeoutMs?: number
  readonly maxBatchChars?: number
  readonly credentials?: RequestCredentials
  readonly maxSessions?: number
}

const DEEPL_PROXY_HOST = '927251094806098001.discordsays.com'
const DEEPL_UPSTREAM_HOST = 'ita-free.www.deepl.com'
const DEFAULT_ENDPOINT = `https://${DEEPL_PROXY_HOST}/cdn/c/${DEEPL_UPSTREAM_HOST}/v2`
const DEEPL_COMMON_UPSTREAM_HEADERS = {
  'Accept-Language': 'en-US,en;q=0.9',
  Origin: 'https://www.deepl.com',
  Referer: 'https://www.deepl.com/',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
}
const DEEPL_START_SESSION_UPSTREAM_HEADERS = {
  ...DEEPL_COMMON_UPSTREAM_HEADERS,
  Accept: 'application/x-protobuf',
  'Content-Type': 'application/x-protobuf',
}
const DEEPL_NEGOTIATE_UPSTREAM_HEADERS = {
  ...DEEPL_COMMON_UPSTREAM_HEADERS,
  Accept: 'application/json, text/plain, */*',
}
const DEEPL_WEBSOCKET_UPSTREAM_HEADERS = {
  ...DEEPL_COMMON_UPSTREAM_HEADERS,
}
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000
const DEFAULT_MAX_BATCH_CHARS = 1_500
const DEFAULT_MAX_SESSIONS = 8
const textEncoder = new TextEncoder()
const textDecoder = new TextDecoder()

export class DeepLTranslateWeb implements Translator {
  readonly name = 'deepl-translate-web'
  private readonly endpoint: string
  private readonly requestTimeoutMs: number
  private readonly maxBatchChars: number
  private readonly credentials: RequestCredentials
  private readonly limiter: AsyncLimiter
  private sessions: DeepLSignalRSession[] = []
  private readonly busy = new Set<DeepLSignalRSession>()

  constructor(options: DeepLTranslateWebOptions = {}) {
    this.endpoint = options.endpoint ?? DEFAULT_ENDPOINT
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
    this.maxBatchChars = options.maxBatchChars ?? DEFAULT_MAX_BATCH_CHARS
    this.credentials = options.credentials ?? 'same-origin'
    this.limiter = new AsyncLimiter(options.maxSessions ?? DEFAULT_MAX_SESSIONS)
  }

  async translateUnits({ units, sourceLang, targetLang, signal }: Parameters<Translator['translateUnits']>[0]): Promise<readonly TranslatedUnit[]> {
    const batches = batchUnits(units, this.maxBatchChars)
    const resultBatches = new Array<readonly string[]>(batches.length)

    await Promise.all(batches.map((batch, bi) =>
      this.limiter.run(async () => {
        if (batch.every(unit => !unit.sourceText.trim())) {
          resultBatches[bi] = batch.map(() => '')
          return
        }
        const session = await this.acquireSession(signal)
        try {
          const translated = await session.translate(serializeBatch(batch), sourceLang, targetLang, signal)
          resultBatches[bi] = parseTranslatedBatch(translated, batch.length)
        } finally {
          this.busy.delete(session)
        }
      }),
    ))

    const out: TranslatedUnit[] = []
    for (let bi = 0; bi < batches.length; bi++) {
      const batch = batches[bi]!
      const results = resultBatches[bi]!
      for (let j = 0; j < batch.length; j++) {
        out.push(toTranslatedUnit(batch[j]!, results[j]))
      }
    }
    return out
  }

  async close(): Promise<void> {
    const sessions = this.sessions.splice(0)
    this.busy.clear()
    for (const session of sessions) session.close()
  }

  private async acquireSession(signal?: AbortSignal): Promise<DeepLSignalRSession> {
    const idle = this.sessions.find(s => !s.isClosed && !this.busy.has(s))
    if (idle) {
      this.busy.add(idle)
      return idle
    }
    for (let i = this.sessions.length - 1; i >= 0; i--) {
      if (this.sessions[i]!.isClosed) this.sessions.splice(i, 1)
    }
    if (this.sessions.length < this.limiter.concurrency) {
      const session = new DeepLSignalRSession({
        endpoint: this.endpoint,
        requestTimeoutMs: this.requestTimeoutMs,
        credentials: this.credentials,
      })
      this.sessions.push(session)
      this.busy.add(session)
      try {
        await session.connect(signal)
      } catch (error) {
        this.busy.delete(session)
        const index = this.sessions.indexOf(session)
        if (index !== -1) this.sessions.splice(index, 1)
        session.close()
        throw error
      }
      return session
    }
    throw new Error('no idle deep session available')
  }
}

// ─── DeepL SignalR Session ────────────────────────────────────────────────

interface Deferred<T> {
  readonly promise: Promise<T>
  readonly resolve: (value: T) => void
  readonly reject: (error: unknown) => void
}

class DeepLSignalRSession {
  private readonly endpoint: string
  private readonly requestTimeoutMs: number
  private readonly credentials: RequestCredentials
  private ws: WebSocket | null = null
  private ready: Deferred<void> | null = null
  private pendingTranslation: Deferred<void> | null = null
  private handshaked = false
  private closed = false
  private baseVersion = 0
  private sourceText = ''
  private targetText = ''
  private maxTextLength = 1_500

  constructor(options: { endpoint: string; requestTimeoutMs: number; credentials: RequestCredentials }) {
    this.endpoint = options.endpoint
    this.requestTimeoutMs = options.requestTimeoutMs
    this.credentials = options.credentials
  }

  get isClosed(): boolean {
    return this.closed
  }

  async connect(signal?: AbortSignal): Promise<void> {
    this.ready = createDeferred<void>()
    try {
      const startSession = await this.startSession(signal)
      const negotiate = await this.negotiate(startSession.signalrEndpoint, signal)
      const ws = new WebSocket(this.websocketUrl(startSession.signalrEndpoint, negotiate.connectionToken))
      this.ws = ws
      ws.binaryType = 'arraybuffer'
      ws.addEventListener('message', event => {
        void this.handleMessage(event.data).catch(error => this.fail(asError(error)))
      })
      ws.addEventListener('close', () => {
        if (!this.closed) this.fail(new Error('deepl websocket closed'))
      })
      ws.addEventListener('error', () => {
        if (!this.closed) this.fail(new Error('deepl websocket failed'))
      })
      await waitForWebSocketOpen(ws, signal, this.requestTimeoutMs)
      ws.send('{"protocol":"messagepack","version":1}\x1e')
      await waitFor(this.ready.promise, signal, this.requestTimeoutMs, 'deepl session timed out')
    } catch (error) {
      this.fail(asError(error))
      throw error
    }
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.pendingTranslation?.reject(new Error('deepl session closed'))
    this.ready?.reject(new Error('deepl session closed'))
    const ws = this.ws
    this.ws = null
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) ws.close()
  }

  async translate(text: string, sourceLang: string | null, targetLang: string, signal?: AbortSignal): Promise<string> {
    if (this.closed) throw new Error('deepl session is closed')
    if (text.length >= this.maxTextLength) {
      throw new Error(`deepl text must not exceed ${this.maxTextLength} characters`)
    }

    const pending = createDeferred<void>()
    this.pendingTranslation = pending
    const currentLen = this.sourceText.length
    this.sourceText = text
    this.send(encodeSignalRMessages([[
      1, {}, '0', 'AppendRequest',
      [new MsgpackExt(4, encodeParticipantRequest({
        baseVersion: this.baseVersion,
        events: translationEvents({ text, currentLen, sourceLang, targetLang }),
      }))],
    ]]))

    try {
      await waitFor(pending.promise, signal, this.requestTimeoutMs, 'deepl translation timed out')
      return this.targetText
    } catch (error) {
      this.fail(asError(error))
      throw error
    } finally {
      if (this.pendingTranslation === pending) this.pendingTranslation = null
    }
  }

  private async startSession(signal?: AbortSignal): Promise<StartSessionResponse> {
    const abort = requestAbortSignal(signal, this.requestTimeoutMs)
    try {
      const res = await fetch(apiUrl(this.endpoint, '/startSession'), {
        method: 'POST',
        headers: proxyRequestHeaders(
          { 'content-type': 'application/x-protobuf' },
          DEEPL_START_SESSION_UPSTREAM_HEADERS,
        ),
        body: encodeStartSessionRequest(),
        signal: abort.signal,
        credentials: this.credentials,
      })
      if (!res.ok) throw new Error(`deepl startSession failed: ${res.status}`)
      return decodeStartSessionResponse(new Uint8Array(await res.arrayBuffer()))
    } finally {
      abort.cleanup()
    }
  }

  private async negotiate(signalrEndpoint: string, signal?: AbortSignal): Promise<NegotiateResponse> {
    const abort = requestAbortSignal(signal, this.requestTimeoutMs)
    try {
      const res = await fetch(this.sessionUrl('/sessions/negotiate', signalrEndpoint, 'negotiateVersion=1'), {
        method: 'POST',
        headers: proxyRequestHeaders({}, DEEPL_NEGOTIATE_UPSTREAM_HEADERS),
        signal: abort.signal,
        credentials: this.credentials,
      })
      if (!res.ok) throw new Error(`deepl negotiate failed: ${res.status}`)
      const payload = await res.json() as Partial<NegotiateResponse>
      if (!payload.connectionToken) throw new Error('deepl negotiate response missing connectionToken')
      return { connectionToken: payload.connectionToken }
    } finally {
      abort.cleanup()
    }
  }

  private websocketUrl(signalrEndpoint: string, connectionToken: string): string {
    const url = this.sessionUrl('/sessions', signalrEndpoint, `id=${encodeURIComponent(connectionToken)}`)
    url.searchParams.set('_h', encodeHeaderBlob(DEEPL_WEBSOCKET_UPSTREAM_HEADERS))
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    return url.href
  }

  private sessionUrl(path: string, signalrEndpoint: string, suffix: string): URL {
    const query = queryString(signalrEndpoint)
    const url = apiUrl(this.endpoint, path)
    url.search = query ? `${query}&${suffix}` : suffix
    return url
  }

  private async handleMessage(data: string | ArrayBuffer | Blob): Promise<void> {
    const bytes = await messageBytes(data)
    if (!this.handshaked) {
      const handshake = JSON.parse(textDecoder.decode(bytes).replace(/\x1e$/, '')) as Record<string, unknown>
      if (Object.keys(handshake).length !== 0) throw new Error('deepl handshake failed')
      this.handshaked = true
      this.send(encodeSignalRMessages([[6]]))
      this.sendParticipate()
      return
    }
    for (const message of decodeSignalRMessages(bytes)) this.handleSignalRMessage(message)
  }

  private sendParticipate(): void {
    this.send(encodeSignalRMessages([[
      1, {}, '1', 'Participate',
      [new MsgpackExt(3, new Uint8Array())],
    ]]))
  }

  private handleSignalRMessage(message: MsgpackValue): void {
    if (!Array.isArray(message)) return
    const type = numberValue(message[0])
    if (type === 1) {
      const target = typeof message[3] === 'string' ? message[3] : ''
      const args = Array.isArray(message[4]) ? message[4] : []
      if (target === 'AppendResponse') {
        args.forEach(arg => this.handleExt(arg))
        this.send(encodeSignalRMessages([[5, {}, '1']]))
      } else if (target === 'OnError') {
        const error = args.map(arg => this.errorFromExt(arg)).find(Boolean) ?? 'deepl returned an error'
        this.fail(new Error(error))
      }
      return
    }
    if (type === 2) { this.handleExt(message[3]); return }
    if (type === 3) {
      const resultType = numberValue(message[3])
      if (resultType === 1) this.fail(new Error(String(message[4] ?? 'deepl invocation failed')))
      if (resultType === 3) this.handleExt(message[4])
      return
    }
    if (type === 6) this.send(encodeSignalRMessages([[6]]))
  }

  private handleExt(value: MsgpackValue | undefined): void {
    if (!(value instanceof MsgpackExt)) return
    if (value.code === 5) {
      this.handleParticipantResponse(decodeParticipantResponse(value.data))
    } else if (value.code === 6) {
      this.fail(new Error(decodeClientErrorInfo(value.data) ?? 'deepl client error'))
    }
  }

  private errorFromExt(value: MsgpackValue | undefined): string | null {
    if (!(value instanceof MsgpackExt) || value.code !== 6) return null
    return decodeClientErrorInfo(value.data)
  }

  private handleParticipantResponse(response: ParticipantResponse): void {
    if (response.confirmedVersion !== null) this.baseVersion = response.confirmedVersion
    if (response.publishedVersion !== null) this.baseVersion = response.publishedVersion
    for (const event of response.events) this.applyFieldEvent(event)
    if (response.idleVersion !== null) {
      this.baseVersion = response.idleVersion
      this.pendingTranslation?.resolve()
    }
    if (response.initialized) this.ready?.resolve()
  }

  private applyFieldEvent(event: DecodedFieldEvent): void {
    if (event.textChange) {
      if (event.fieldName === 2 || event.fieldName === 6) {
        this.targetText = applyTextChange(this.targetText, event.textChange)
      }
    }
    if (event.setProperty?.maxTextLength) this.maxTextLength = event.setProperty.maxTextLength
  }

  private send(payload: Uint8Array): void {
    if (this.closed) return
    const ws = this.ws
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error('deepl websocket is not open')
    ws.send(payload)
  }

  private fail(error: Error): void {
    if (this.closed) return
    this.closed = true
    this.ready?.reject(error)
    this.pendingTranslation?.reject(error)
    const ws = this.ws
    this.ws = null
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) ws.close()
  }
}

// ─── Translation events ───────────────────────────────────────────────────

const PARTICIPANT_ID = 2

interface FieldEventInput {
  readonly fieldName: number
  readonly participantId: number
  readonly textChange?: { readonly end: number; readonly text: string }
  readonly setProperty?: SetPropertyInput
}

interface SetPropertyInput {
  readonly propertyName: number
  readonly requestedSourceLanguage?: string
  readonly requestedTargetLanguage?: string
  readonly calculatedTargetLanguage?: string
  readonly maxTextLength?: number
}

function translationEvents(input: { text: string; currentLen: number; sourceLang: string | null; targetLang: string }): readonly FieldEventInput[] {
  return [
    { fieldName: 2, participantId: PARTICIPANT_ID, setProperty: { propertyName: 5, requestedTargetLanguage: input.targetLang } },
    input.sourceLang
      ? { fieldName: 1, participantId: PARTICIPANT_ID, setProperty: { propertyName: 3, requestedSourceLanguage: input.sourceLang } }
      : { fieldName: 1, participantId: PARTICIPANT_ID, setProperty: { propertyName: 3 } },
    { fieldName: 1, participantId: PARTICIPANT_ID, textChange: { end: input.currentLen, text: input.text } },
  ]
}

// ─── Protobuf encoding ────────────────────────────────────────────────────

interface StartSessionResponse { readonly signalrEndpoint: string }
interface NegotiateResponse { readonly connectionToken: string }

function encodeStartSessionRequest(): Uint8Array {
  return protoMessageBytes([
    protoVarint(1, 1),
    protoMessage(2, protoMessageBytes([
      protoMessage(1, encodeDocumentField({ fieldName: 1, properties: [{ propertyName: 14, maxTextLength: 1_500 }] })),
      protoMessage(1, encodeDocumentField({
        fieldName: 2,
        properties: [
          { propertyName: 5, requestedTargetLanguage: 'en-US' },
          { propertyName: 18, calculatedTargetLanguage: 'en-US' },
        ],
      })),
    ])),
    protoMessage(3, protoVarint(2, 1)),
  ])
}

function encodeDocumentField(input: { fieldName: number; properties: readonly SetPropertyInput[] }): Uint8Array {
  return protoMessageBytes([
    protoVarint(1, input.fieldName),
    ...input.properties.map(p => protoMessage(4, encodeSetPropertyOperation(p))),
  ])
}

function encodeParticipantRequest(input: { baseVersion: number; events: readonly FieldEventInput[] }): Uint8Array {
  return protoMessage(1, protoMessageBytes([
    ...input.events.map(e => protoMessage(1, encodeFieldEvent(e))),
    protoMessage(2, encodeEventVersion(input.baseVersion)),
  ]))
}

function encodeFieldEvent(input: FieldEventInput): Uint8Array {
  const fields = [
    protoVarint(1, input.fieldName),
    input.textChange ? protoMessage(2, protoMessageBytes([
      protoMessage(1, protoVarint(2, input.textChange.end)),
      protoString(2, input.textChange.text),
    ])) : null,
    input.setProperty ? protoMessage(5, encodeSetPropertyOperation(input.setProperty)) : null,
    protoMessage(6, protoVarint(1, input.participantId)),
  ]
  return protoMessageBytes(fields.filter(f => f !== null))
}

function encodeSetPropertyOperation(input: SetPropertyInput): Uint8Array {
  const fields: Uint8Array[] = [protoVarint(1, input.propertyName)]
  if (input.requestedSourceLanguage) fields.push(protoMessage(4, protoMessage(1, protoString(1, input.requestedSourceLanguage))))
  if (input.requestedTargetLanguage) fields.push(protoMessage(5, protoMessage(1, protoString(1, input.requestedTargetLanguage))))
  if (input.maxTextLength) fields.push(protoMessage(14, protoVarint(1, input.maxTextLength)))
  if (input.calculatedTargetLanguage) fields.push(protoMessage(18, protoMessage(1, protoString(1, input.calculatedTargetLanguage))))
  return protoMessageBytes(fields)
}

function encodeEventVersion(version: number): Uint8Array {
  return protoMessage(1, protoVarint(1, version))
}

// ─── Protobuf decoding ────────────────────────────────────────────────────

interface ParticipantResponse {
  readonly initialized: boolean
  readonly confirmedVersion: number | null
  readonly publishedVersion: number | null
  readonly idleVersion: number | null
  readonly events: readonly DecodedFieldEvent[]
}

interface DecodedFieldEvent {
  readonly fieldName: number
  readonly textChange: DecodedTextChange | null
  readonly setProperty: DecodedSetProperty | null
}

interface DecodedTextChange { readonly start: number; readonly end: number; readonly text: string }
interface DecodedSetProperty { readonly propertyName: number; readonly maxTextLength: number | null }
interface ProtoField { readonly fieldNumber: number; readonly wireType: number; readonly value: number | Uint8Array }

function decodeStartSessionResponse(bytes: Uint8Array): StartSessionResponse {
  const fields = readProtoFields(bytes)
  const signalrEndpoint = firstString(fields, 5)
  if (!signalrEndpoint) throw new Error('deepl startSession response missing signalrEndpoint')
  return { signalrEndpoint }
}

function decodeParticipantResponse(bytes: Uint8Array): ParticipantResponse {
  const fields = readProtoFields(bytes)
  const confirmedVersion = firstMessage(fields, 1, decodeConfirmedVersion) ?? null
  const published = firstMessage(fields, 3, decodePublishedMessage)
  const idleVersion = firstMessage(fields, 4, decodeMetaInfoIdleVersion) ?? null
  return {
    initialized: fieldMessages(fields, 2).length > 0,
    confirmedVersion,
    publishedVersion: published?.version ?? null,
    idleVersion,
    events: published?.events ?? [],
  }
}

function decodeConfirmedVersion(bytes: Uint8Array): number | null {
  return firstMessage(readProtoFields(bytes), 1, decodeEventVersion) ?? null
}

function decodePublishedMessage(bytes: Uint8Array): { readonly version: number | null; readonly events: readonly DecodedFieldEvent[] } {
  const fields = readProtoFields(bytes)
  return {
    version: firstMessage(fields, 2, decodeEventVersion) ?? null,
    events: fieldMessages(fields, 1).map(decodeFieldEvent),
  }
}

function decodeMetaInfoIdleVersion(bytes: Uint8Array): number | null {
  return firstMessage(readProtoFields(bytes), 1, idle =>
    firstMessage(readProtoFields(idle), 1, decodeEventVersion) ?? null) ?? null
}

function decodeEventVersion(bytes: Uint8Array): number | null {
  return firstMessage(readProtoFields(bytes), 1, v => firstVarint(readProtoFields(v), 1)) ?? null
}

function decodeFieldEvent(bytes: Uint8Array): DecodedFieldEvent {
  const fields = readProtoFields(bytes)
  return {
    fieldName: firstVarint(fields, 1) ?? 0,
    textChange: firstMessage(fields, 2, decodeTextChange) ?? null,
    setProperty: firstMessage(fields, 5, decodeSetProperty) ?? null,
  }
}

function decodeTextChange(bytes: Uint8Array): DecodedTextChange {
  const fields = readProtoFields(bytes)
  const range = firstMessage(fields, 1, rangeBytes => {
    const rangeFields = readProtoFields(rangeBytes)
    return { start: firstVarint(rangeFields, 1) ?? 0, end: firstVarint(rangeFields, 2) ?? 0 }
  }) ?? { start: 0, end: 0 }
  return { start: range.start, end: range.end, text: firstString(fields, 2) ?? '' }
}

function decodeSetProperty(bytes: Uint8Array): DecodedSetProperty {
  const fields = readProtoFields(bytes)
  return {
    propertyName: firstVarint(fields, 1) ?? 0,
    maxTextLength: firstMessage(fields, 14, maxBytes => firstVarint(readProtoFields(maxBytes), 1)) ?? null,
  }
}

function decodeClientErrorInfo(bytes: Uint8Array): string | null {
  const fields = readProtoFields(bytes)
  return firstMessage(fields, 2, detail => firstString(readProtoFields(detail), 1)) ?? null
}

function firstMessage<T>(fields: readonly ProtoField[], fieldNumber: number, decode: (bytes: Uint8Array) => T): T | null {
  const message = fieldMessages(fields, fieldNumber)[0]
  return message ? decode(message) : null
}

function fieldMessages(fields: readonly ProtoField[], fieldNumber: number): Uint8Array[] {
  return fields.filter(f => f.fieldNumber === fieldNumber && f.value instanceof Uint8Array).map(f => f.value as Uint8Array)
}

function firstString(fields: readonly ProtoField[], fieldNumber: number): string | null {
  const value = fieldMessages(fields, fieldNumber)[0]
  return value ? textDecoder.decode(value) : null
}

function firstVarint(fields: readonly ProtoField[], fieldNumber: number): number | null {
  const field = fields.find(item => item.fieldNumber === fieldNumber && typeof item.value === 'number')
  return typeof field?.value === 'number' ? field.value : null
}

function readProtoFields(bytes: Uint8Array): ProtoField[] {
  const fields: ProtoField[] = []
  let offset = 0
  while (offset < bytes.length) {
    const key = readVarint(bytes, offset)
    offset = key.offset
    const fieldNumber = key.value >> 3
    const wireType = key.value & 7
    if (wireType === 0) {
      const value = readVarint(bytes, offset)
      offset = value.offset
      fields.push({ fieldNumber, wireType, value: value.value })
    } else if (wireType === 1) {
      offset += 8
    } else if (wireType === 2) {
      const length = readVarint(bytes, offset)
      offset = length.offset
      const end = offset + length.value
      fields.push({ fieldNumber, wireType, value: bytes.slice(offset, end) })
      offset = end
    } else if (wireType === 5) {
      offset += 4
    } else {
      throw new Error(`unsupported protobuf wire type ${wireType}`)
    }
  }
  return fields
}

function applyTextChange(original: string, change: DecodedTextChange): string {
  return `${original.slice(0, change.start)}${change.text}${original.slice(change.end)}`
}

// ─── MessagePack encoding/decoding ────────────────────────────────────────

type MsgpackValue = null | boolean | number | string | Uint8Array | MsgpackExt | MsgpackValue[] | { readonly [key: string]: MsgpackValue }

class MsgpackExt {
  readonly code: number
  readonly data: Uint8Array
  constructor(code: number, data: Uint8Array) { this.code = code; this.data = data }
}

function encodeSignalRMessages(messages: readonly MsgpackValue[]): Uint8Array {
  const parts: Uint8Array[] = []
  for (const message of messages) {
    const body = encodeMsgpack(message)
    parts.push(encodeVarint(body.length), body)
  }
  return concatBytes(parts)
}

function decodeSignalRMessages(bytes: Uint8Array): MsgpackValue[] {
  const messages: MsgpackValue[] = []
  let offset = 0
  while (offset < bytes.length) {
    const size = readVarint(bytes, offset)
    offset = size.offset
    const end = offset + size.value
    messages.push(decodeMsgpack(bytes.slice(offset, end)).value)
    offset = end
  }
  return messages
}

function encodeMsgpack(value: MsgpackValue): Uint8Array {
  if (value === null) return byte(0xc0)
  if (typeof value === 'boolean') return byte(value ? 0xc3 : 0xc2)
  if (typeof value === 'number') return encodeMsgpackNumber(value)
  if (typeof value === 'string') return encodeMsgpackString(value)
  if (value instanceof Uint8Array) return encodeMsgpackBytes(value)
  if (value instanceof MsgpackExt) return encodeMsgpackExt(value)
  if (Array.isArray(value)) return encodeMsgpackArray(value)
  return encodeMsgpackMap(value)
}

function encodeMsgpackNumber(value: number): Uint8Array {
  if (!Number.isInteger(value)) return concatBytes([byte(0xcb), float64Bytes(value)])
  if (value >= 0 && value < 0x80) return byte(value)
  if (value >= 0 && value <= 0xff) return concatBytes([byte(0xcc), byte(value)])
  if (value >= 0 && value <= 0xffff) return concatBytes([byte(0xcd), uint16Bytes(value)])
  if (value >= 0) return concatBytes([byte(0xce), uint32Bytes(value)])
  if (value >= -32) return byte(0xe0 | (value + 32))
  if (value >= -128) return concatBytes([byte(0xd0), byte(value & 0xff)])
  if (value >= -32768) return concatBytes([byte(0xd1), int16Bytes(value)])
  return concatBytes([byte(0xd2), int32Bytes(value)])
}

function encodeMsgpackString(value: string): Uint8Array {
  const bytes = textEncoder.encode(value)
  if (bytes.length < 32) return concatBytes([byte(0xa0 | bytes.length), bytes])
  if (bytes.length <= 0xff) return concatBytes([byte(0xd9), byte(bytes.length), bytes])
  if (bytes.length <= 0xffff) return concatBytes([byte(0xda), uint16Bytes(bytes.length), bytes])
  return concatBytes([byte(0xdb), uint32Bytes(bytes.length), bytes])
}

function encodeMsgpackBytes(value: Uint8Array): Uint8Array {
  if (value.length <= 0xff) return concatBytes([byte(0xc4), byte(value.length), value])
  if (value.length <= 0xffff) return concatBytes([byte(0xc5), uint16Bytes(value.length), value])
  return concatBytes([byte(0xc6), uint32Bytes(value.length), value])
}

function encodeMsgpackExt(value: MsgpackExt): Uint8Array {
  if (value.data.length <= 0xff) return concatBytes([byte(0xc7), byte(value.data.length), byte(value.code), value.data])
  if (value.data.length <= 0xffff) return concatBytes([byte(0xc8), uint16Bytes(value.data.length), byte(value.code), value.data])
  return concatBytes([byte(0xc9), uint32Bytes(value.data.length), byte(value.code), value.data])
}

function encodeMsgpackArray(value: readonly MsgpackValue[]): Uint8Array {
  const body = concatBytes(value.map(encodeMsgpack))
  if (value.length < 16) return concatBytes([byte(0x90 | value.length), body])
  if (value.length <= 0xffff) return concatBytes([byte(0xdc), uint16Bytes(value.length), body])
  return concatBytes([byte(0xdd), uint32Bytes(value.length), body])
}

function encodeMsgpackMap(value: { readonly [key: string]: MsgpackValue }): Uint8Array {
  const entries = Object.entries(value)
  const body = concatBytes(entries.flatMap(([k, v]) => [encodeMsgpackString(k), encodeMsgpack(v)]))
  if (entries.length < 16) return concatBytes([byte(0x80 | entries.length), body])
  if (entries.length <= 0xffff) return concatBytes([byte(0xde), uint16Bytes(entries.length), body])
  return concatBytes([byte(0xdf), uint32Bytes(entries.length), body])
}

function decodeMsgpack(bytes: Uint8Array, offset = 0): { readonly value: MsgpackValue; readonly offset: number } {
  const prefix = bytes[offset]
  if (prefix === undefined) throw new Error('unexpected end of msgpack data')
  offset += 1
  if (prefix <= 0x7f) return { value: prefix, offset }
  if (prefix >= 0x80 && prefix <= 0x8f) return decodeMsgpackMap(bytes, offset, prefix & 0x0f)
  if (prefix >= 0x90 && prefix <= 0x9f) return decodeMsgpackArray(bytes, offset, prefix & 0x0f)
  if (prefix >= 0xa0 && prefix <= 0xbf) return decodeMsgpackString(bytes, offset, prefix & 0x1f)
  if (prefix >= 0xe0) return { value: prefix - 0x100, offset }
  if (prefix === 0xc0) return { value: null, offset }
  if (prefix === 0xc2 || prefix === 0xc3) return { value: prefix === 0xc3, offset }
  if (prefix === 0xc4) return decodeMsgpackBin(bytes, offset, bytes[offset] ?? 0, 1)
  if (prefix === 0xc5) return decodeMsgpackBin(bytes, offset + 2, readUint16(bytes, offset), 0)
  if (prefix === 0xc6) return decodeMsgpackBin(bytes, offset + 4, readUint32(bytes, offset), 0)
  if (prefix === 0xc7) return decodeMsgpackExt(bytes, offset + 2, bytes[offset] ?? 0, bytes[offset + 1] ?? 0)
  if (prefix === 0xc8) return decodeMsgpackExt(bytes, offset + 3, readUint16(bytes, offset), bytes[offset + 2] ?? 0)
  if (prefix === 0xc9) return decodeMsgpackExt(bytes, offset + 5, readUint32(bytes, offset), bytes[offset + 4] ?? 0)
  if (prefix === 0xca) return { value: readFloat32(bytes, offset), offset: offset + 4 }
  if (prefix === 0xcb) return { value: readFloat64(bytes, offset), offset: offset + 8 }
  if (prefix === 0xcc) return { value: bytes[offset] ?? 0, offset: offset + 1 }
  if (prefix === 0xcd) return { value: readUint16(bytes, offset), offset: offset + 2 }
  if (prefix === 0xce) return { value: readUint32(bytes, offset), offset: offset + 4 }
  if (prefix === 0xd0) return { value: int8(bytes[offset] ?? 0), offset: offset + 1 }
  if (prefix === 0xd1) return { value: readInt16(bytes, offset), offset: offset + 2 }
  if (prefix === 0xd2) return { value: readInt32(bytes, offset), offset: offset + 4 }
  if (prefix >= 0xd4 && prefix <= 0xd8) {
    const lengths = [1, 2, 4, 8, 16]
    return decodeMsgpackExt(bytes, offset + 1, lengths[prefix - 0xd4]!, bytes[offset] ?? 0)
  }
  if (prefix === 0xd9) return decodeMsgpackString(bytes, offset + 1, bytes[offset] ?? 0)
  if (prefix === 0xda) return decodeMsgpackString(bytes, offset + 2, readUint16(bytes, offset))
  if (prefix === 0xdb) return decodeMsgpackString(bytes, offset + 4, readUint32(bytes, offset))
  if (prefix === 0xdc) return decodeMsgpackArray(bytes, offset + 2, readUint16(bytes, offset))
  if (prefix === 0xdd) return decodeMsgpackArray(bytes, offset + 4, readUint32(bytes, offset))
  if (prefix === 0xde) return decodeMsgpackMap(bytes, offset + 2, readUint16(bytes, offset))
  if (prefix === 0xdf) return decodeMsgpackMap(bytes, offset + 4, readUint32(bytes, offset))
  throw new Error(`unsupported msgpack prefix ${prefix}`)
}

function decodeMsgpackString(bytes: Uint8Array, offset: number, length: number): { readonly value: string; readonly offset: number } {
  return { value: textDecoder.decode(bytes.slice(offset, offset + length)), offset: offset + length }
}

function decodeMsgpackBin(bytes: Uint8Array, offset: number, length: number, lengthBytes: number): { readonly value: Uint8Array; readonly offset: number } {
  const start = offset + lengthBytes
  return { value: bytes.slice(start, start + length), offset: start + length }
}

function decodeMsgpackExt(bytes: Uint8Array, offset: number, length: number, code: number): { readonly value: MsgpackExt; readonly offset: number } {
  return { value: new MsgpackExt(int8(code), bytes.slice(offset, offset + length)), offset: offset + length }
}

function decodeMsgpackArray(bytes: Uint8Array, offset: number, length: number): { readonly value: MsgpackValue[]; readonly offset: number } {
  const value: MsgpackValue[] = []
  let nextOffset = offset
  for (let i = 0; i < length; i++) {
    const item = decodeMsgpack(bytes, nextOffset)
    value.push(item.value)
    nextOffset = item.offset
  }
  return { value, offset: nextOffset }
}

function decodeMsgpackMap(bytes: Uint8Array, offset: number, length: number): { readonly value: { readonly [key: string]: MsgpackValue }; readonly offset: number } {
  const value: Record<string, MsgpackValue> = {}
  let nextOffset = offset
  for (let i = 0; i < length; i++) {
    const key = decodeMsgpack(bytes, nextOffset)
    const item = decodeMsgpack(bytes, key.offset)
    value[String(key.value)] = item.value
    nextOffset = item.offset
  }
  return { value, offset: nextOffset }
}

// ─── Varint / binary helpers ──────────────────────────────────────────────

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = []
  let next = Math.max(0, Math.floor(value))
  do {
    let current = next & 0x7f
    next = Math.floor(next / 128)
    if (next) current |= 0x80
    bytes.push(current)
  } while (next)
  return new Uint8Array(bytes)
}

function readVarint(bytes: Uint8Array, offset: number): { readonly value: number; readonly offset: number } {
  let result = 0
  let shift = 0
  let nextOffset = offset
  while (nextOffset < bytes.length) {
    const current = bytes[nextOffset++]!
    result += (current & 0x7f) * 2 ** shift
    if ((current & 0x80) === 0) return { value: result, offset: nextOffset }
    shift += 7
  }
  throw new Error('incomplete varint')
}

function numberValue(value: MsgpackValue | undefined): number {
  return typeof value === 'number' ? value : -1
}

function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const part of parts) { out.set(part, offset); offset += part.length }
  return out
}

function byte(value: number): Uint8Array { return new Uint8Array([value & 0xff]) }
function uint16Bytes(value: number): Uint8Array { return new Uint8Array([(value >> 8) & 0xff, value & 0xff]) }
function uint32Bytes(value: number): Uint8Array { return new Uint8Array([(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff]) }
function int16Bytes(value: number): Uint8Array { return uint16Bytes(value & 0xffff) }
function int32Bytes(value: number): Uint8Array { return uint32Bytes(value >>> 0) }
function float64Bytes(value: number): Uint8Array { const bytes = new Uint8Array(8); new DataView(bytes.buffer).setFloat64(0, value); return bytes }

function readUint16(bytes: Uint8Array, offset: number): number { return ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0) }
function readUint32(bytes: Uint8Array, offset: number): number { return ((bytes[offset] ?? 0) * 2 ** 24) + ((bytes[offset + 1] ?? 0) << 16) + ((bytes[offset + 2] ?? 0) << 8) + (bytes[offset + 3] ?? 0) }
function readInt16(bytes: Uint8Array, offset: number): number { const v = readUint16(bytes, offset); return v & 0x8000 ? v - 0x10000 : v }
function readInt32(bytes: Uint8Array, offset: number): number { return readUint32(bytes, offset) | 0 }
function readFloat32(bytes: Uint8Array, offset: number): number { return new DataView(bytes.buffer, bytes.byteOffset + offset, 4).getFloat32(0) }
function readFloat64(bytes: Uint8Array, offset: number): number { return new DataView(bytes.buffer, bytes.byteOffset + offset, 8).getFloat64(0) }
function int8(value: number): number { return value & 0x80 ? value - 0x100 : value }

function protoMessageBytes(fields: readonly Uint8Array[]): Uint8Array { return concatBytes(fields) }
function protoVarint(fieldNumber: number, value: number): Uint8Array { return concatBytes([encodeVarint((fieldNumber << 3) | 0), encodeVarint(value)]) }
function protoString(fieldNumber: number, value: string): Uint8Array { return protoBytes(fieldNumber, textEncoder.encode(value)) }
function protoMessage(fieldNumber: number, value: Uint8Array): Uint8Array { return protoBytes(fieldNumber, value) }
function protoBytes(fieldNumber: number, value: Uint8Array): Uint8Array { return concatBytes([encodeVarint((fieldNumber << 3) | 2), encodeVarint(value.length), value]) }

function proxyRequestHeaders(headers: Record<string, string>, upstreamHeaders: Record<string, string>): Headers {
  const out = new Headers(headers)
  if (Object.keys(upstreamHeaders).length > 0) out.set('X-Proxy-Headers', encodeHeaderBlob(upstreamHeaders))
  return out
}

function encodeHeaderBlob(headers: Record<string, string>): string {
  const sorted: Record<string, string> = {}
  for (const key of Object.keys(headers).sort()) sorted[key] = headers[key]!
  const bytes = textEncoder.encode(JSON.stringify(sorted))
  let bin = ''
  for (const byteValue of bytes) bin += String.fromCharCode(byteValue)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function apiUrl(endpoint: string, path: string): URL {
  const url = new URL(endpoint, baseHref())
  url.pathname = `${url.pathname.replace(/\/$/, '')}${path}`
  url.search = ''
  return url
}

function queryString(url: string): string { const q = url.indexOf('?'); return q === -1 ? '' : url.slice(q + 1) }
function baseHref(): string { return typeof document !== 'undefined' ? document.baseURI : typeof location !== 'undefined' ? location.href : 'http://localhost/' }

async function messageBytes(data: string | ArrayBuffer | Blob): Promise<Uint8Array> {
  if (typeof data === 'string') return textEncoder.encode(data)
  if (data instanceof Blob) return new Uint8Array(await data.arrayBuffer())
  return new Uint8Array(data)
}

// ─── Async helpers ────────────────────────────────────────────────────────

function requestAbortSignal(signal: AbortSignal | undefined, timeoutMs: number): { readonly signal: AbortSignal; readonly cleanup: () => void } {
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  const timeout = timeoutMs > 0 ? setTimeout(() => controller.abort(new Error('request timed out')), timeoutMs) : null
  if (signal?.aborted) abort()
  else signal?.addEventListener('abort', abort, { once: true })
  return {
    signal: controller.signal,
    cleanup: () => {
      if (timeout !== null) clearTimeout(timeout)
      signal?.removeEventListener('abort', abort)
    },
  }
}

function waitFor<T>(promise: Promise<T>, signal: AbortSignal | undefined, timeoutMs: number, timeoutMessage: string): Promise<T> {
  if (!signal && timeoutMs <= 0) return promise
  return new Promise<T>((resolve, reject) => {
    let settled = false
    const cleanup = () => { settled = true; if (timeout !== null) clearTimeout(timeout); signal?.removeEventListener('abort', abort) }
    const abort = () => { if (settled) return; cleanup(); reject(signal?.reason ?? new Error('operation aborted')) }
    const timeout = timeoutMs > 0 ? setTimeout(() => { if (settled) return; cleanup(); reject(new Error(timeoutMessage)) }, timeoutMs) : null
    if (signal?.aborted) { abort(); return }
    signal?.addEventListener('abort', abort, { once: true })
    promise.then(value => { if (settled) return; cleanup(); resolve(value) }, error => { if (settled) return; cleanup(); reject(error) })
  })
}

function waitForWebSocketOpen(ws: WebSocket, signal: AbortSignal | undefined, timeoutMs: number): Promise<void> {
  return waitFor(new Promise<void>((resolve, reject) => {
    const cleanup = () => { ws.removeEventListener('open', open); ws.removeEventListener('error', error); ws.removeEventListener('close', close) }
    const open = () => { cleanup(); resolve() }
    const error = () => { cleanup(); reject(new Error('deepl websocket failed to open')) }
    const close = () => { cleanup(); reject(new Error('deepl websocket closed before open')) }
    ws.addEventListener('open', open, { once: true })
    ws.addEventListener('error', error, { once: true })
    ws.addEventListener('close', close, { once: true })
  }), signal, timeoutMs, 'deepl websocket open timed out')
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

function asError(error: unknown): Error { return error instanceof Error ? error : new Error(String(error)) }
