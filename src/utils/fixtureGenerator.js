const fs = require("node:fs/promises");
const path = require("node:path");
const sharp = require("sharp");

const WIDTH = 640;
const HEIGHT = 480;

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

function makeBuffer(pixelWriter) {
  const buffer = Buffer.alloc(WIDTH * HEIGHT * 3);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const idx = (y * WIDTH + x) * 3;
      const [r, g, b] = pixelWriter(x, y);
      buffer[idx] = r;
      buffer[idx + 1] = g;
      buffer[idx + 2] = b;
    }
  }

  return buffer;
}

function withNoise(value, amount) {
  const noise = (Math.random() * 2 - 1) * amount;
  return Math.max(0, Math.min(255, Math.round(value + noise)));
}

async function writeJpeg(rawBuffer, outPath) {
  await sharp(rawBuffer, { raw: { width: WIDTH, height: HEIGHT, channels: 3 } })
    .jpeg({ quality: 92 })
    .toFile(outPath);
}

async function generateCopper(outPath) {
  const raw = makeBuffer((x, y) => {
    const gradX = x / WIDTH;
    const gradY = y / HEIGHT;
    const r = withNoise(160 + gradX * 70 + gradY * 20, 20);
    const g = withNoise(90 + gradX * 35, 14);
    const b = withNoise(50 + gradY * 15, 10);
    return [r, g, b];
  });

  await writeJpeg(raw, outPath);
}

async function generateIron(outPath) {
  const raw = makeBuffer((x, y) => {
    const stripe = ((x + y) % 40) < 20 ? 1 : -1;
    const base = 120 + stripe * 12;
    const value = withNoise(base, 24);
    return [value, value - 4, value + 3];
  });

  await writeJpeg(raw, outPath);
}

async function generatePlastic(outPath) {
  const raw = makeBuffer((x, y) => {
    const grad = x / WIDTH;
    const r = withNoise(25 + grad * 35, 10);
    const g = withNoise(80 + (y / HEIGHT) * 40, 12);
    const b = withNoise(170 + grad * 60, 20);
    return [r, g, b];
  });

  await writeJpeg(raw, outPath);
}

async function generateBlurry(sourcePath, outPath) {
  await sharp(sourcePath).blur(14).jpeg({ quality: 88 }).toFile(outPath);
}

async function generateMixed(copperPath, ironPath, outPath) {
  const left = await sharp(copperPath).resize(Math.floor(WIDTH / 2), HEIGHT).toBuffer();
  const right = await sharp(ironPath).resize(Math.ceil(WIDTH / 2), HEIGHT).toBuffer();

  await sharp({ create: { width: WIDTH, height: HEIGHT, channels: 3, background: { r: 90, g: 90, b: 90 } } })
    .composite([
      { input: left, left: 0, top: 0 },
      { input: right, left: Math.floor(WIDTH / 2), top: 0 }
    ])
    .jpeg({ quality: 92 })
    .toFile(outPath);
}

async function ensureRealImages(baseDir) {
  await ensureDir(baseDir);

  const copperPath = path.join(baseDir, "copper.jpg");
  const ironPath = path.join(baseDir, "iron.jpg");
  const plasticPath = path.join(baseDir, "plastic.jpg");
  const blurryPath = path.join(baseDir, "blurry.jpg");
  const mixedPath = path.join(baseDir, "mixed.jpg");

  await generateCopper(copperPath);
  await generateIron(ironPath);
  await generatePlastic(plasticPath);
  await generateBlurry(copperPath, blurryPath);
  await generateMixed(copperPath, ironPath, mixedPath);

  return {
    copperPath,
    ironPath,
    plasticPath,
    blurryPath,
    mixedPath
  };
}

module.exports = {
  ensureRealImages
};
