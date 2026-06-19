import * as Comlink from 'comlink'
import { WorkerVisionImpl } from './WorkerVisionImpl'

Comlink.expose(new WorkerVisionImpl())
