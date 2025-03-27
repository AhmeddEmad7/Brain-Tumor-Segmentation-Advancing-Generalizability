/**
 * Scans an HTMLCanvasElement for non-black pixels and returns a new canvas
 * cropped to the bounding box of non-black areas.
 * @param sourceCanvas - The original canvas to crop
 * @param threshold - How close to black a pixel can be (0-255).
 *                    E.g., threshold=10 means (R,G,B) all < 10 is considered black.
 */
export default function cropBlack(sourceCanvas: HTMLCanvasElement, threshold = 5): HTMLCanvasElement {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const ctx = sourceCanvas.getContext('2d');
    if (!ctx) {
      return sourceCanvas; // Fallback, no context
    }
  
    // Get pixel data
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
  
    let xMin = width, yMin = height;
    let xMax = 0, yMax = 0;
  
    // Scan each pixel
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const a = data[idx + 3]; // alpha channel
  
        // If the pixel is sufficiently non-black and not transparent
        const isNonBlack = (r > threshold || g > threshold || b > threshold) && a > 0;
        if (isNonBlack) {
          if (x < xMin) xMin = x;
          if (x > xMax) xMax = x;
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }
      }
    }
  
    // If everything is black, xMax < xMin, etc.
    if (xMax < xMin || yMax < yMin) {
      // No non-black pixels found, return original
      return sourceCanvas;
    }
  
    // Crop the found bounding box
    const cropWidth = xMax - xMin + 1;
    const cropHeight = yMax - yMin + 1;
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = cropWidth;
    croppedCanvas.height = cropHeight;
    const croppedCtx = croppedCanvas.getContext('2d');
    if (!croppedCtx) {
      return sourceCanvas;
    }
  
    // Extract sub-region from original
    const subImageData = ctx.getImageData(xMin, yMin, cropWidth, cropHeight);
    croppedCtx.putImageData(subImageData, 0, 0);
  
    return croppedCanvas;
  }
  