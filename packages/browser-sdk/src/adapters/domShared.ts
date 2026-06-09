export interface DomPageOptions {
  readonly imageSelector?: string
  readonly hostSelector?: string
}

export function imagesIn(container: HTMLElement, imageSelector = 'img'): HTMLImageElement[] {
  return [...container.querySelectorAll<HTMLImageElement>(imageSelector)]
}

export function imageAt(container: HTMLElement, index: number, imageSelector = 'img'): HTMLImageElement {
  const image = imagesIn(container, imageSelector)[index]
  if (!image) throw new RangeError(`reader image index out of range: ${index}`)
  return image
}

export function pageHostForImage(image: HTMLImageElement, hostSelector: string | undefined, index: number): HTMLElement {
  if (!hostSelector) return ensureImageHost(image)
  const host = image.closest<HTMLElement>(hostSelector)
  if (!host) throw new Error(`reader host not found for image index: ${index}`)
  assertHostMatchesImage(host, image, index)
  return host
}

function ensureImageHost(image: HTMLImageElement): HTMLElement {
  const parent = image.parentElement
  if (parent?.dataset.typoonDomReaderHost === 'true') return parent
  if (!parent) return image

  const style = getComputedStyle(image)
  const wrapper = document.createElement(style.display === 'block' ? 'div' : 'span')
  wrapper.dataset.typoonDomReaderHost = 'true'
  wrapper.style.display = style.display === 'block' ? 'block' : 'inline-block'
  wrapper.style.position = 'relative'
  wrapper.style.lineHeight = '0'
  parent.insertBefore(wrapper, image)
  wrapper.appendChild(image)
  return wrapper
}

function assertHostMatchesImage(host: HTMLElement, image: HTMLImageElement, index: number): void {
  const hostRect = host.getBoundingClientRect()
  const imageRect = image.getBoundingClientRect()
  const leftMatches = Math.abs(hostRect.left - imageRect.left) <= 1
  const topMatches = Math.abs(hostRect.top - imageRect.top) <= 1
  const widthMatches = Math.abs(hostRect.width - imageRect.width) <= 1
  const heightMatches = Math.abs(hostRect.height - imageRect.height) <= 1
  if (!leftMatches || !topMatches || !widthMatches || !heightMatches) {
    throw new Error(`reader host must match rendered image bounds for image index: ${index}`)
  }
}
