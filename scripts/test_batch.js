const fs = require('node:fs');
const path = require('node:path');

async function testBatch() {
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

  console.log("Starting Batch Testing...");
  console.log("=======================\n");

  let totalCorrect = 0;
  let totalImages = 0;
  
  for (const folder of folders) {
    console.log(`[${folder.toUpperCase()}]`);
    const folderPath = path.join(testImagesDir, folder);
    const files = fs.readdirSync(folderPath).filter(f => f.match(/\.(jpg|jpeg|png)$/i));
    
    let folderCorrect = 0;
    
    for (const file of files) {
      const filePath = path.join(folderPath, file);
      totalImages++;
      
      try {
        const fileBuffer = fs.readFileSync(filePath);
        const fileBlob = new Blob([fileBuffer], { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('image', fileBlob, file);
        
        const response = await fetch('http://localhost:3000/analyze', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          console.log(`${file} → API Error ${response.status} ❌`);
          continue;
        }
        
        const result = await response.json();
        const predicted = result.material;
        const confidence = result.confidence;
        const source = result.decision?.source || 'unknown';
        
        const folderLower = folder.toLowerCase();
        let isCorrect = (predicted === folderLower);
        if ((folderLower === 'metel_iron' || folderLower === 'metal_iron') && predicted === 'metal') {
          isCorrect = true;
        }
        if (isCorrect) {
          folderCorrect++;
          totalCorrect++;
        }
        
        const icon = isCorrect ? '✅' : '❌';
        console.log(`${file} → ${predicted} (${confidence}) [src: ${source}] ${icon}`);
        
      } catch (err) {
        console.log(`${file} → Request Failed: ${err.message} ❌`);
      }
    }
    
    const accuracy = files.length > 0 ? ((folderCorrect / files.length) * 100).toFixed(1) : 0;
    console.log(`\nAccuracy: ${accuracy}%\n`);
  }

  console.log("=======================");
  const overallAccuracy = totalImages > 0 ? ((totalCorrect / totalImages) * 100).toFixed(1) : 0;
  console.log(`Overall Accuracy: ${overallAccuracy}%`);
}

testBatch();
