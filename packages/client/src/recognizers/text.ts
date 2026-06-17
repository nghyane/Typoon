import type { ImagePixels } from '../domain/image'
import type { RecognizedTextPage } from '../domain/text'

export interface TextRecognitionOptions {
  readonly sourceLang: string | null
  readonly pageIndex: number
  readonly signal?: AbortSignal
}

export interface TextRecognizer {
  readonly name: string
  recognizeText(image: ImagePixels, options: TextRecognitionOptions): Promise<RecognizedTextPage>
}
