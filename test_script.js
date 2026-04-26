const fs = require("node:fs");
const modelService = require("./src/services/modelService.js");

async function run() {
  const imagePath = "tests/real_images/copper.jpg";
  const imageBuffer = fs.readFileSync(imagePath);
  
  // Hijack console.warn to capture the exact message
  const originalWarn = console.warn;
  console.warn = (...args) => {
    originalWarn(...args);
    if (args[0].includes("efficientnet_lite0")) {
      process.stderr.write("CAPTURED_ERROR: " + args.join(" ") + "\n");
    }
  };

  try {
    await modelService.init();
    await modelService.infer(imageBuffer);
  } catch (error) {
    // Ignore
  }
}

run();
