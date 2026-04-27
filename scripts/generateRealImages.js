const path = require("node:path");
const { ensureRealImages } = require("../tests/support/fixtureGenerator");

async function main() {
  const outDir = path.resolve(__dirname, "../tests/real_images");
  const files = await ensureRealImages(outDir);
  process.stdout.write(`generated fixtures in ${outDir}\n`);
  process.stdout.write(`${JSON.stringify(files, null, 2)}\n`);
}

main().catch((error) => {
  console.error("Failed to generate image fixtures", error);
  process.exit(1);
});
