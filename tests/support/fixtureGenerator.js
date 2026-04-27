const fs = require("node:fs/promises");
const path = require("node:path");
const sharp = require("sharp");

const WIDTH = 640;
const HEIGHT = 480;

function clampChannel(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function seededNoise(x, y, salt, amount) {
  const seed = Math.sin((x + 1) * 12.9898 + (y + 1) * 78.233 + salt * 19.17) * 43758.5453;
  return ((seed - Math.floor(seed)) * 2 - 1) * amount;
}

function buildRawBuffer(pixelWriter) {
  const buffer = Buffer.alloc(WIDTH * HEIGHT * 3);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const index = (y * WIDTH + x) * 3;
      const [red, green, blue] = pixelWriter(x, y);
      buffer[index] = clampChannel(red);
      buffer[index + 1] = clampChannel(green);
      buffer[index + 2] = clampChannel(blue);
    }
  }

  return buffer;
}

async function writeJpeg(rawBuffer, outputPath) {
  await sharp(rawBuffer, {
    raw: {
      width: WIDTH,
      height: HEIGHT,
      channels: 3
    }
  })
    .jpeg({ quality: 90 })
    .toFile(outputPath);
}

async function createCopper(outputPath) {
  const raw = buildRawBuffer((x, y) => {
    const normalizedX = x / WIDTH;
    const normalizedY = y / HEIGHT;
    const red = 150 + (normalizedX * 80) + seededNoise(x, y, 1, 28);
    const green = 82 + (normalizedY * 35) + seededNoise(x, y, 2, 16);
    const blue = 38 + seededNoise(x, y, 3, 12);
    return [red, green, blue];
  });

  await writeJpeg(raw, outputPath);
}

async function createPlastic(outputPath) {
  const raw = buildRawBuffer((x, y) => {
    const normalizedX = x / WIDTH;
    const normalizedY = y / HEIGHT;
    const red = 22 + (normalizedX * 24) + seededNoise(x, y, 4, 10);
    const green = 120 + (normalizedY * 50) + seededNoise(x, y, 5, 12);
    const blue = 170 + (normalizedX * 55) + seededNoise(x, y, 6, 14);
    return [red, green, blue];
  });

  await writeJpeg(raw, outputPath);
}

async function createIron(outputPath) {
  const raw = buildRawBuffer((x, y) => {
    const stripe = ((x + y) % 36) < 18 ? 1 : -1;
    const base = 120 + (stripe * 15);
    const red = base + seededNoise(x, y, 7, 18);
    const green = base + seededNoise(x, y, 8, 18);
    const blue = base + 6 + seededNoise(x, y, 9, 16);
    return [red, green, blue];
  });

  await writeJpeg(raw, outputPath);
}

async function createMixed(copperPath, plasticPath, outputPath) {
  const left = await sharp(copperPath).resize(Math.floor(WIDTH / 2), HEIGHT).toBuffer();
  const right = await sharp(plasticPath).resize(Math.ceil(WIDTH / 2), HEIGHT).toBuffer();

  await sharp({
    create: {
      width: WIDTH,
      height: HEIGHT,
      channels: 3,
      background: { r: 90, g: 90, b: 90 }
    }
  })
    .composite([
      { input: left, left: 0, top: 0 },
      { input: right, left: Math.floor(WIDTH / 2), top: 0 }
    ])
    .jpeg({ quality: 90 })
    .toFile(outputPath);
}

async function createBlurry(sourcePath, outputPath) {
  await sharp(sourcePath).blur(16).jpeg({ quality: 88 }).toFile(outputPath);
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function ensureFile(filePath, factory) {
  try {
    await fs.access(filePath);
  } catch {
    await factory(filePath);
  }
}

async function ensureRealImages(baseDir) {
  await ensureDir(baseDir);

  const copperPath = path.join(baseDir, "copper.jpg");
  const plasticPath = path.join(baseDir, "plastic.jpg");
  const ironPath = path.join(baseDir, "iron.jpg");
  const blurryPath = path.join(baseDir, "blurry.jpg");
  const mixedPath = path.join(baseDir, "mixed.jpg");

  await ensureFile(copperPath, createCopper);
  await ensureFile(plasticPath, createPlastic);
  await ensureFile(ironPath, createIron);
  await ensureFile(blurryPath, (outputPath) => createBlurry(copperPath, outputPath));
  await ensureFile(mixedPath, (outputPath) => createMixed(copperPath, plasticPath, outputPath));

  return {
    copperPath,
    plasticPath,
    ironPath,
    blurryPath,
    mixedPath
  };
}

module.exports = {
  ensureRealImages
};