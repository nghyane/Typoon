import type { CapabilityProgress, CapabilityStatus, StatusListener, Unsubscribe } from '../domain/capability'

export class CapabilityMachine {
  private current: CapabilityStatus
  private readonly name: string
  private readonly listeners = new Set<StatusListener>()

  constructor(name: string) {
    this.name = name
    this.current = { name, state: 'idle' }
  }

  status(): CapabilityStatus {
    return this.current
  }

  subscribe(listener: StatusListener): Unsubscribe {
    this.listeners.add(listener)
    listener(this.current)
    return () => this.listeners.delete(listener)
  }

  resolving(): void {
    this.publish({ name: this.name, state: 'resolving' })
  }

  downloading(progress: CapabilityProgress): void {
    this.publish({ name: this.name, state: 'downloading', progress })
  }

  initializing(): void {
    this.publish({ name: this.name, state: 'initializing' })
  }

  ready(): void {
    this.publish({ name: this.name, state: 'ready' })
  }

  failed(error: unknown): void {
    this.publish({ name: this.name, state: 'failed', error })
  }

  mirror(status: CapabilityStatus): void {
    switch (status.state) {
      case 'idle': return
      case 'resolving': return this.resolving()
      case 'downloading': return this.downloading(status.progress)
      case 'initializing': return this.initializing()
      case 'ready': return this.ready()
      case 'failed': return this.failed(status.error)
    }
  }

  private publish(status: CapabilityStatus): void {
    this.current = status
    for (const listener of this.listeners) listener(status)
  }
}
