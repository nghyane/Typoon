import type { ModelAssetCache } from './modelTypes'

const DB_NAME = 'typoon-models'
const STORE_NAME = 'models'
const DB_VERSION = 1

export class ModelIndexedDBCache implements ModelAssetCache {
  private dbPromise: Promise<IDBDatabase> | null = null

  private db(): Promise<IDBDatabase> {
    this.dbPromise ??= new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = () => {
        const db = req.result
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME)
        }
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
    return this.dbPromise
  }

  async match(key: string): Promise<ArrayBuffer | null> {
    try {
      const db = await this.db()
      return new Promise<ArrayBuffer | null>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly')
        const req = tx.objectStore(STORE_NAME).get(key)
        req.onsuccess = () => resolve(req.result ?? null)
        req.onerror = () => reject(req.error)
      })
    } catch {
      return null
    }
  }

  async put(key: string, bytes: ArrayBuffer): Promise<void> {
    try {
      const db = await this.db()
      return new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite')
        const req = tx.objectStore(STORE_NAME).put(bytes, key)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      })
    } catch {
      // IndexedDB unavailable (private browsing, quota, etc.) — silent fail
    }
  }

  async keys(): Promise<string[]> {
    try {
      const db = await this.db()
      return new Promise<string[]>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly')
        const req = tx.objectStore(STORE_NAME).getAllKeys()
        req.onsuccess = () => resolve((req.result as IDBValidKey[]).map(String))
        req.onerror = () => reject(req.error)
      })
    } catch {
      return []
    }
  }

  async delete(key: string): Promise<void> {
    try {
      const db = await this.db()
      return new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite')
        const req = tx.objectStore(STORE_NAME).delete(key)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      })
    } catch {
      // IndexedDB unavailable — silent fail
    }
  }
}
