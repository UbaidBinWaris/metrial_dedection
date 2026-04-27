# Material Detection Microservice

CPU-only Node.js microservice for local image analysis with singleton model loading, multi-model fusion, and no external APIs.

## Folder Structure

```text
metrial_dedection/
├── mobilenet_v2_1.0_224.tflite
├── models/
│   ├── efficientnet-lite0.tar.gz
│   ├── ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1.tar.gz
│   └── ssd_mobilenet_v2_fpnlite_320x320.tflite   # optional runtime detector asset
├── scripts/
│   └── generateRealImages.js
├── src/
│   ├── app.js
│   ├── server.js
│   ├── models/
│   │   └── modelLoader.js
│   ├── pipeline/
│   │   ├── blurService.js
│   │   ├── classificationService.js
│   │   ├── detectionService.js
│   │   ├── featureService.js
│   │   ├── fusionService.js
│   │   ├── lightService.js
│   │   └── weightService.js
│   └── routes/
│       └── analyze.js
└── tests/
    ├── analyze.integration.test.js
    ├── real_images/
    └── support/
        └── fixtureGenerator.js
```

## Output Shape

```json
{
  "accepted": true,
  "material": "copper",
  "confidence": 0.82,
  "alternatives": [
    { "material": "aluminum", "confidence": 0.41 }
  ],
  "blur": {
    "isBlurry": false,
    "score": 248.11
  },
  "lighting": {
    "brightness": 0.57,
    "isTooDark": false,
    "isTooBright": false,
    "status": "good"
  },
  "objects": [
    { "label": "bottle", "score": 0.87, "box": { "top": 0.1, "left": 0.2, "bottom": 0.8, "right": 0.6 } }
  ],
  "classifications": [
    { "classId": 901, "label": "imagenet_901", "score": 0.14 }
  ],
  "features": {
    "brightness": 0.57,
    "variance": 0.018,
    "metallicScore": 0.62
  },
  "warnings": [],
  "weight": 0.914,
  "decision": {
    "source": "detection",
    "reasons": ["object:bottle"]
  }
}
```

## Runtime Notes

- Classification uses the local MobileNet and EfficientNet Lite models via TensorFlow Lite.
- Detection uses `models/ssd_mobilenet_v2_fpnlite_320x320.tflite` if that file is present.
- The repository currently contains an SSD SavedModel archive, not the direct `.tflite` detector file, so the detection stage degrades safely to an empty object list until the TFLite asset is added.
- Models are loaded once at startup and reused for all requests.

## Run

```bash
npm install
npm run generate:fixtures
npm start
```

## Test

```bash
npm test
```

## Example curl

```bash
curl -X POST http://localhost:3000/analyze \
  -F "image=@tests/real_images/copper.jpg"
```
