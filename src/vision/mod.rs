pub mod border;
pub mod detection;
pub mod inpaint;
pub mod ocr;

/// Convert an RGB image to NCHW f32 tensor with per-channel normalization:
///   output[c] = (pixel[c] / 255.0 - mean[c]) / std[c]
///
/// Processes the flat pixel buffer in a single pass (HWC → NCHW),
/// avoiding per-pixel `get_pixel` overhead.
///
/// `content_h`/`content_w` may be smaller than the output tensor dimensions
/// to support zero-padded inputs (e.g. PP-OCR det pads to multiples of 32).
pub fn rgb_to_nchw(
    rgb: &image::RgbImage,
    out_h: usize,
    out_w: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> ndarray::Array4<f32> {
    let content_h = (rgb.height() as usize).min(out_h);
    let content_w = (rgb.width() as usize).min(out_w);
    let img_w = rgb.width() as usize;

    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, out_h, out_w));
    let raw = rgb.as_raw();

    // Write directly into contiguous channel slices for better cache behavior.
    for c in 0..3usize {
        let m = mean[c];
        let s_inv = 1.0 / std[c];
        let mut plane = arr.slice_mut(ndarray::s![0, c, .., ..]);
        for y in 0..content_h {
            let row_off = y * img_w * 3;
            for x in 0..content_w {
                plane[[y, x]] = (raw[row_off + x * 3 + c] as f32 / 255.0 - m) * s_inv;
            }
        }
    }

    arr
}
