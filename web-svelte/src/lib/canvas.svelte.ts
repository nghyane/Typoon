// $lib/canvas.svelte.ts
// Canvas display — draws a Blob onto <canvas>, keeps ImageBitmap alive.
// Use with: <canvas use:draw={blob} />

export function draw(node: HTMLCanvasElement, blob: Blob | null) {
  let bitmap: ImageBitmap | null = null;
  let cancelled = false;

  async function render(b: Blob | null) {
    bitmap?.close();
    bitmap = null;

    if (!b) {
      node.width = 0;
      node.height = 0;
      return;
    }

    try {
      const bmp = await createImageBitmap(b);
      if (cancelled) { bmp.close(); return; }

      node.width = bmp.width;
      node.height = bmp.height;
      const ctx = node.getContext('2d');
      if (ctx) ctx.drawImage(bmp, 0, 0);
      bitmap = bmp;
    } catch {
      // blob decode failed — silently skip
    }
  }

  render(blob);

  return {
    update(b: Blob | null) { render(b); },
    destroy() {
      cancelled = true;
      bitmap?.close();
    },
  };
}
