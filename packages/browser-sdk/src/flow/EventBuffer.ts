export class EventBuffer<T> {
  private readonly listeners = new Set<(event: T) => void>()
  private readonly queue: T[] = []
  private readonly waiters: ((value: IteratorResult<T>) => void)[] = []
  private closed = false

  subscribe(listener: (event: T) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  events(): AsyncIterable<T> {
    return { [Symbol.asyncIterator]: () => ({ next: () => this.next() }) }
  }

  emit(event: T): void {
    if (this.closed) return
    for (const listener of this.listeners) listener(event)
    const waiter = this.waiters.shift()
    if (waiter) waiter({ value: event, done: false })
    else this.queue.push(event)
  }

  close(): void {
    this.closed = true
    while (this.waiters.length) this.waiters.shift()!({ value: undefined, done: true })
  }

  private next(): Promise<IteratorResult<T>> {
    const event = this.queue.shift()
    if (event) return Promise.resolve({ value: event, done: false })
    if (this.closed) return Promise.resolve({ value: undefined, done: true })
    return new Promise(resolve => this.waiters.push(resolve))
  }
}
