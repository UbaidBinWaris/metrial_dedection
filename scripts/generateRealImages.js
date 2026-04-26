const path = require("node:path");
const { ensureRealImages } = require("../src/utils/fixtureGenerator");

async function main() {
  const outDir = path.resolve(__dirname, "../tests/real_images");
  const files = await ensureRealImages(outDir);
  console.log("Generated real image fixtures:", files);
}

main().catch((error) => {
  console.error("Failed to generate image fixtures", error);
  process.exit(1);
});
