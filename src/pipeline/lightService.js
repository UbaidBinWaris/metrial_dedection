const sharp = require("sharp");

const LIGHT_IMAGE_SIZE = 192;
const TOO_DARK_THRESHOLD = 0.22;
const TOO_BRIGHT_THRESHOLD = 0.85;

async function analyze(imageBuffer) {
  const { data } = await sharp(imageBuffer)
    .rotate()
    .resize(LIGHT_IMAGE_SIZE, LIGHT_IMAGE_SIZE, { fit: "inside", withoutEnlargement: true })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  let sum = 0;

  for (const value of data) {
    sum += value / 255;
  }

  const brightness = data.length === 0 ? 0 : sum / data.length;
  const isTooDark = brightness < TOO_DARK_THRESHOLD;
  const isTooBright = brightness > TOO_BRIGHT_THRESHOLD;

  return {
    brightness: Number(brightness.toFixed(4)),
    isTooDark,
    isTooBright,
    status: isTooDark ? "too_dark" : isTooBright ? "too_bright" : "good"
  };
}

module.exports = {
  analyze
};