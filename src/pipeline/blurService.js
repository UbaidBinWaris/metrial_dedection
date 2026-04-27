const sharp = require("sharp");

const BLUR_IMAGE_SIZE = 256;
const BLUR_THRESHOLD = 25;

function variance(values) {
  if (values.length === 0) {
    return 0;
  }

  let sum = 0;
  let sumSquares = 0;

  for (const value of values) {
    sum += value;
    sumSquares += value * value;
  }

  const mean = sum / values.length;
  return (sumSquares / values.length) - (mean * mean);
}

async function analyze(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .rotate()
    .resize(BLUR_IMAGE_SIZE, BLUR_IMAGE_SIZE, { fit: "inside", withoutEnlargement: true })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;
  const responses = [];

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const centerIndex = y * width + x;
      const center = data[centerIndex];
      const left = data[centerIndex - 1];
      const right = data[centerIndex + 1];
      const up = data[centerIndex - width];
      const down = data[centerIndex + width];
      const laplacian = (4 * center) - left - right - up - down;
      responses.push(laplacian);
    }
  }

  const score = variance(responses);

  return {
    isBlurry: score < BLUR_THRESHOLD,
    score: Number(score.toFixed(2))
  };
}

module.exports = {
  analyze
};