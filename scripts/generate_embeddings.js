const fs = require('node:fs');
const path = require('node:path');
const classificationService = require('../src/pipeline/classificationService');

async function generate() {
  const testImagesDir = path.join(__dirname, '../test_images');
  if (!fs.existsSync(testImagesDir)) {
    console.error(`Dataset directory not found: ${testImagesDir}`);
    process.exit(1);
  }

  const folders = fs.readdirSync(testImagesDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name);

  if (folders.length === 0) {
    console.log("No category folders found in test_images/");
    process.exit(0);
  }

  console.log("Starting Embedding Generation...");
  const db = [];

  for (const folder of folders) {
    console.log(`Processing [${folder.toUpperCase()}]...`);
    const folderPath = path.join(testImagesDir, folder);
    const files = fs.readdirSync(folderPath).filter(f => f.match(/\.(jpg|jpeg|png)$/i));
    
    for (const file of files) {
      const filePath = path.join(folderPath, file);
      try {
        const fileBuffer = fs.readFileSync(filePath);
        
        // Extract embedding via existing model
        const result = await classificationService.analyze(fileBuffer);
        const embedding = result.embedding;
        
        if (embedding && embedding.length > 0) {
          db.push({
            label: folder.toLowerCase(),
            filename: file,
            embedding: Array.from(embedding)
          });
          console.log(` ✔ ${file} embedded.`);
        } else {
          console.log(` ❌ ${file} - no embedding returned.`);
        }
      } catch (err) {
        console.log(` ❌ ${file} - Failed: ${err.message}`);
      }
    }
  }

  const configDir = path.join(__dirname, '../src/config');
  if (!fs.existsSync(configDir)) fs.mkdirSync(configDir, { recursive: true });
  
  const outPath = path.join(configDir, 'embeddings.json');
  fs.writeFileSync(outPath, JSON.stringify(db, null, 2));
  console.log(`\nSuccessfully generated embeddings for ${db.length} images and saved to ${outPath}`);
  
  process.exit(0);
}

generate().catch(e => {
  console.error("Fatal error", e);
  process.exit(1);
});
