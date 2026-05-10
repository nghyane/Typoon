// Storage abstraction the rest of `core/` depends on. Plain async
// key/value of JSON-serialisable values. Real implementations live in
// `shell/adapters/` (chrome.storage) and tests can pass an in-memory
// stub.

export interface StorageAdapter {
  get<T>(key: string): Promise<T | undefined>
  set<T>(key: string, value: T): Promise<void>
  del(key: string): Promise<void>
}
