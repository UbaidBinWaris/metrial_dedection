const sharp = require("sharp");

const FEATURE_IMAGE_SIZE = 224;
const HISTOGRAM_BINS = 16;

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function normalizeHistogram(histogram, total) {
  if (total === 0) {
    return histogram.map(() => 0);
  }

  return histogram.map((value) => Number((value / total).toFixed(6)));
}

async function extract(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .rotate()
    .resize(FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE, { fit: "inside", withoutEnlargement: true })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixelCount = info.width * info.height;
  const histogram = {
    red: new Array(HISTOGRAM_BINS).fill(0),
    green: new Array(HISTOGRAM_BINS).fill(0),
    blue: new Array(HISTOGRAM_BINS).fill(0)
  };

  let sumBrightness = 0;
  let sumBrightnessSquares = 0;
  let sumRed = 0;
  let sumGreen = 0;
  let sumBlue = 0;
  let sumSaturation = 0;
  let brightPixelCount = 0;
  let neutralPixelCount = 0;

  for (let index = 0; index < pixelCount; index += 1) {
    const red = data[index * 3] / 255;
    const green = data[(index * 3) + 1] / 255;
    const blue = data[(index * 3) + 2] / 255;

    const luma = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue);
    const maxChannel = Math.max(red, green, blue);
    const minChannel = Math.min(red, green, blue);
    const saturation = maxChannel === 0 ? 0 : (maxChannel - minChannel) / maxChannel;
    const channelSpread = Math.abs(red - green) + Math.abs(green - blue) + Math.abs(red - blue);

    sumBrightness += luma;
    sumBrightnessSquares += luma * luma;
    sumRed += red;
    sumGreen += green;
    sumBlue += blue;
    sumSaturation += saturation;

    if (luma > 0.78) {
      brightPixelCount += 1;
    }

    if (channelSpread < 0.18) {
      neutralPixelCount += 1;
    }

    histogram.red[Math.min(HISTOGRAM_BINS - 1, Math.floor(red * HISTOGRAM_BINS))] += 1;
    histogram.green[Math.min(HISTOGRAM_BINS - 1, Math.floor(green * HISTOGRAM_BINS))] += 1;
    histogram.blue[Math.min(HISTOGRAM_BINS - 1, Math.floor(blue * HISTOGRAM_BINS))] += 1;
  }

  const brightness = pixelCount === 0 ? 0 : sumBrightness / pixelCount;
  const variance = pixelCount === 0 ? 0 : (sumBrightnessSquares / pixelCount) - (brightness * brightness);
  const averageColor = {
    red: pixelCount === 0 ? 0 : Number((sumRed / pixelCount).toFixed(4)),
    green: pixelCount === 0 ? 0 : Number((sumGreen / pixelCount).toFixed(4)),
    blue: pixelCount === 0 ? 0 : Number((sumBlue / pixelCount).toFixed(4))
  };
  const saturation = pixelCount === 0 ? 0 : sumSaturation / pixelCount;
  const highlightRatio = pixelCount === 0 ? 0 : brightPixelCount / pixelCount;
  const neutralRatio = pixelCount === 0 ? 0 : neutralPixelCount / pixelCount;
  const metallicScore = clamp((neutralRatio * 0.45) + ((1 - saturation) * 0.35) + (highlightRatio * 0.2));

  return {
    brightness: Number(brightness.toFixed(4)),
    variance: Number(variance.toFixed(4)),
    colorHistogram: {
      red: normalizeHistogram(histogram.red, pixelCount),
      green: normalizeHistogram(histogram.green, pixelCount),
      blue: normalizeHistogram(histogram.blue, pixelCount)
    },
    averageColor,
    saturation: Number(saturation.toFixed(4)),
    metallicScore: Number(metallicScore.toFixed(4)),
    highlightRatio: Number(highlightRatio.toFixed(4)),
    dimensions: {
      width: info.width,
      height: info.height
    }
  };
}

module.exports = {
  extract
};