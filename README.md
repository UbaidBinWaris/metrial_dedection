# Material Detection Microservice (Node.js + TensorFlow Lite)

Production-ready CPU-only microservice for image material analysis.

## Folder Structure

```text
metrial_dedection/
├── mobilenet_v2_1.0_224.tflite
├── efficientnet-lite0.tar.gz
├── package.json
├── src/
│   ├── app.js
│   ├── server.js
│   ├── config/
│   │   └── materialProfiles.js
│   ├── services/
│   │   ├── modelService.js
│   │   ├── featureService.js
│   │   └── heuristicService.js
│   └── utils/
│       └── fixtureGenerator.js
├── scripts/
│   └── generateRealImages.js
└── tests/
    ├── analyze.integration.test.js
    └── real_images/
        ├── copper.jpg
        ├── iron.jpg
        ├── plastic.jpg
        ├── blurry.jpg
        └── mixed.jpg
```

## API

### POST /analyze

- Content-Type: `multipart/form-data`
- Field name: `image`

Response shape:

```json
{
  "material": "copper",
  "confidence": 0.82,
  "alternatives": [
    { "material": "aluminum", "confidence": 0.12 }
  ],
  "features": {
    "brightness": 0.62,
    "variance": 0.13,
    "colorDistribution": {
      "red": [],
      "green": [],
      "blue": [],
      "meanRGB": { "r": 0.6, "g": 0.4, "b": 0.2 },
      "saturation": 0.19,
      "dominantChannel": "red"
    }
  },
  "reasoning": "High reflectivity and warm tones suggest copper"
}
```

## Run

```bash
npm install
npm run generate:fixtures
node src/server.js
```

Server runs on `http://localhost:3000`.

## Test

```bash
npm test
```

## Example curl

```bash
curl -X POST "http://localhost:3000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@tests/real_images/copper.jpg"
```

## Performance Notes

- Model loading is singleton-based and happens once at startup.
- Inference runs CPU-only via TensorFlow Lite runtime.
- Image processing uses `sharp` with fixed `224x224` input size.
- Memory usage is controlled by processing images in-memory and reusing loaded models.
