import type { Capability, ReadyOptions } from '../../domain/capability'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'

export interface TextRegionRunner extends Capability {
  run(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]>
}
