const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("node:path");
const fs = require("node:fs");

let session = null;

const COCO_LABELS = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
  "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

const MATERIAL_MAP = {
  "bottle": "plastic",
  "cup": "plastic",
  "bowl": "plastic",
  "handbag": "plastic",
  "bag": "plastic",
  "laptop": "metal",
  "microwave": "metal",
  "oven": "metal",
  "toaster": "metal",
  "sink": "metal",
  "refrigerator": "metal",
  "book": "paper",
  "scissors": "metal",
  "cell phone": "mixed"
};

async function getSession() {
  if (session) return session;
  const modelPath = path.join(__dirname, "../../models/yolov8n.onnx");
  
  if (!fs.existsSync(modelPath)) {
    console.error(`[YOLO] Model file NOT found at: ${modelPath}`);
    return null;
  }

  try {
    session = await ort.InferenceSession.create(modelPath);
    console.log("[YOLO] Loaded model successfully");
    return session;
  } catch (error) {
    console.error(`[YOLO] Failed to load model: ${error.message}`);
    return null;
  }
}

function intersectionOverUnion(box1, box2) {
  const x1 = Math.max(box1[0], box2[0]);
  const y1 = Math.max(box1[1], box2[1]);
  const x2 = Math.min(box1[2], box2[2]);
  const y2 = Math.min(box1[3], box2[3]);
  
  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);
  const intersectionArea = width * height;
  
  const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  const unionArea = box1Area + box2Area - intersectionArea;
  
  return intersectionArea / unionArea;
}

function nonMaximumSuppression(boxes, iouThreshold = 0.45) {
  boxes.sort((a, b) => b.score - a.score);
  const result = [];
  while (boxes.length > 0) {
    const box = boxes.shift();
    result.push(box);
    boxes = boxes.filter(item => {
      const iou = intersectionOverUnion(
        [box.box.left, box.box.top, box.box.right, box.box.bottom],
        [item.box.left, item.box.top, item.box.right, item.box.bottom]
      );
      return iou < iouThreshold;
    });
  }
  return result;
}

async function analyze(imageBuffer) {
  const yolosession = await getSession();
  if (!yolosession) {
    return { objects: [], degraded: true, warning: "YOLO model unavailable" };
  }

  try {
    // STEP 3 — PREPROCESS IMAGE (640x640, float32, [1, 3, 640, 640])
    const { data, info } = await sharp(imageBuffer)
      .resize(640, 640, { fit: "fill" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const floatData = new Float32Array(3 * 640 * 640);
    // HWC to CHW conversion and normalization to [0, 1]
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 640 * 640; i++) {
        floatData[c * 640 * 640 + i] = data[i * 3 + c] / 255.0;
      }
    }

    const inputTensor = new ort.Tensor("float32", floatData, [1, 3, 640, 640]);
    const inputName = yolosession.inputNames[0];
    const feeds = { [inputName]: inputTensor };

    // STEP 3 — RUN INFERENCE
    const results = await yolosession.run(feeds);
    
    // Dynamically retrieve output tensor
    const outputName = yolosession.outputNames[0];
    const output = results[outputName].data; // Float32Array [1, 84, 8400]
    
    // STEP 4 — PARSE OUTPUT
    const numClasses = 80;
    const numBoxes = 8400;
    const candidates = [];

    for (let i = 0; i < numBoxes; i++) {
      let maxScore = -1;
      let classId = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = output[(4 + c) * numBoxes + i];
        if (score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      if (maxScore > 0.4) {
        const cx = output[0 * numBoxes + i];
        const cy = output[1 * numBoxes + i];
        const w = output[2 * numBoxes + i];
        const h = output[3 * numBoxes + i];

        const x1 = (cx - w / 2) / 640;
        const y1 = (cy - h / 2) / 640;
        const x2 = (cx + w / 2) / 640;
        const y2 = (cy + h / 2) / 640;

        candidates.push({
          label: COCO_LABELS[classId],
          score: maxScore,
          box: {
            top: Math.max(0, y1),
            left: Math.max(0, x1),
            bottom: Math.min(1, y2),
            right: Math.min(1, x2)
          }
        });
      }
    }

    const filteredObjects = nonMaximumSuppression(candidates);
    const averageScore = filteredObjects.length > 0 
      ? filteredObjects.reduce((sum, obj) => sum + obj.score, 0) / filteredObjects.length 
      : 0;

    console.log(`[YOLO] Found ${filteredObjects.length} objects`);
    if (filteredObjects.length > 0) {
      console.log(`[YOLO] Top object: ${filteredObjects[0].label} (${filteredObjects[0].score.toFixed(2)})`);
    }

    return {
      objects: filteredObjects,
      detectionConfidence: averageScore,
      degraded: false
    };
  } catch (error) {
    console.error(`[YOLO] Inference failed: ${error.message}`);
    return { objects: [], degraded: true, error: error.message };
  }
}

module.exports = {
  analyze
};