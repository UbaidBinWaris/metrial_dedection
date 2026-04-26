const sharp = require("sharp");

const IMAGE_SIZE = 224;
const HISTOGRAM_BINS = 16;

function dominantChannel(meanRGB) {
  if (meanRGB.r >= meanRGB.g && meanRGB.r >= meanRGB.b) {
    return "red";
  }
  if (meanRGB.g >= meanRGB.r && meanRGB.g >= meanRGB.b) {
    return "green";
  }
  return "blue";
}

function normalizeHistogram(values, total) {
  if (total <= 0) {
    return values.map(() => 0);
  }
  return values.map((value) => value / total);
}

class FeatureService {
  async extractFeatures(imageBuffer) {
    const { data } = await sharp(imageBuffer)
      .resize(IMAGE_SIZE, IMAGE_SIZE, { fit: "cover" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelCount = IMAGE_SIZE * IMAGE_SIZE;

    let sumLuma = 0;
    let sumLumaSquared = 0;
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    let sumSaturation = 0;

    const redHist = new Array(HISTOGRAM_BINS).fill(0);
    const greenHist = new Array(HISTOGRAM_BINS).fill(0);
    const blueHist = new Array(HISTOGRAM_BINS).fill(0);

    for (let i = 0; i < pixelCount; i += 1) {
      const r = data[i * 3] / 255;
      const g = data[i * 3 + 1] / 255;
      const b = data[i * 3 + 2] / 255;

      sumR += r;
      sumG += g;
      sumB += b;

      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      sumLuma += luma;
      sumLumaSquared += luma * luma;

      const maxRGB = Math.max(r, g, b);
      const minRGB = Math.min(r, g, b);
      const sat = maxRGB === 0 ? 0 : (maxRGB - minRGB) / maxRGB;
      sumSaturation += sat;

      redHist[Math.min(HISTOGRAM_BINS - 1, Math.floor(r * HISTOGRAM_BINS))] += 1;
      greenHist[Math.min(HISTOGRAM_BINS - 1, Math.floor(g * HISTOGRAM_BINS))] += 1;
      blueHist[Math.min(HISTOGRAM_BINS - 1, Math.floor(b * HISTOGRAM_BINS))] += 1;
    }

    const brightness = sumLuma / pixelCount;
    const variance = Math.sqrt(Math.max(0, sumLumaSquared / pixelCount - brightness * brightness));

    const meanRGB = {
      r: sumR / pixelCount,
      g: sumG / pixelCount,
      b: sumB / pixelCount
    };

    return {
      brightness,
      variance,
      colorDistribution: {
        red: normalizeHistogram(redHist, pixelCount),
        green: normalizeHistogram(greenHist, pixelCount),
        blue: normalizeHistogram(blueHist, pixelCount),
        meanRGB,
        saturation: sumSaturation / pixelCount,
        dominantChannel: dominantChannel(meanRGB)
      }
    };
  }
}

module.exports = new FeatureService();
