const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { Interpreter } = require("node-tflite");

const ROOT_DIR = path.resolve(__dirname, "../..");
const MODEL_DIR = path.join(ROOT_DIR, "models");

const CLASSIFIER_CONFIGS = [
  {
    name: "mobilenet_v2",
    modelPath: path.join(ROOT_DIR, "mobilenet_v2_1.0_224.tflite"),
    inputSize: 224,
    normalization: "signed",
    backgroundOffset: 1
  },
  {
    name: "efficientnet_lite0",
    modelPath: path.join(ROOT_DIR, "efficientnet-lite0", "efficientnet-lite0-fp32.tflite"),
    fallbackModelPath: path.join(ROOT_DIR, "efficientnet-lite0", "efficientnet-lite0-int8.tflite"),
    archivePath: path.join(MODEL_DIR, "efficientnet-lite0.tar.gz"),
    inputSize: 224,
    normalization: "signed",
    backgroundOffset: 0
  }
];

const DETECTOR_CONFIG = {
  name: "ssd_mobilenet_v2_fpnlite_320x320",
  modelPath: path.join(MODEL_DIR, "ssd_mobilenet_v2_fpnlite_320x320.tflite"),
  archivePath: path.join(MODEL_DIR, "ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1.tar.gz"),
  inputSize: 320
};

function fileExists(filePath) {
  return fsp.access(filePath).then(() => true).catch(() => false);
}

function getTensorSize(tensor) {
  const dims = Array.isArray(tensor?.dims) ? tensor.dims : [];
  if (dims.length === 0) {
    return 0;
  }

  return dims.reduce((total, value) => total * value, 1);
}

function loadInterpreter(modelPath) {
  const bytes = fs.readFileSync(modelPath);
  const interpreter = new Interpreter(bytes);
  interpreter.allocateTensors();
  return interpreter;
}

class ModelLoader {
  constructor() {
    this._initPromise = null;
    this._loadCount = 0;
    this._classifiers = [];
    this._detector = null;
    this._warnings = [];
  }

  async initialize() {
    if (this._initPromise) {
      return this._initPromise;
    }

    this._initPromise = this._loadModels();
    return this._initPromise;
  }

  getLoadCount() {
    return this._loadCount;
  }

  getWarnings() {
    return [...this._warnings];
  }

  getClassifierModels() {
    return this._classifiers;
  }

  getDetectorModel() {
    return this._detector;
  }

  async _ensureEfficientNetExtracted(config) {
    if (await fileExists(config.modelPath)) {
      return;
    }

    if (!(await fileExists(config.archivePath))) {
      return;
    }

    execFileSync("tar", ["-xzf", config.archivePath], {
      cwd: ROOT_DIR,
      stdio: "ignore"
    });
  }

  _buildClassifierRecord(config, resolvedPath) {
    const interpreter = loadInterpreter(resolvedPath);
    let inputType = "float32";
    let outputSize = 1000;

    try {
      const inputTensor = interpreter.inputs[0];
      const outputTensor = interpreter.outputs[0];
      inputType = String(inputTensor?.type || "float32").toLowerCase();
      outputSize = getTensorSize(outputTensor);
    } catch (e) {
      console.warn(`[Loader] Could not determine metadata for classifier ${config.name}, using defaults`);
    }

    return {
      name: config.name,
      filePath: resolvedPath,
      interpreter,
      inputSize: config.inputSize,
      normalization: config.normalization,
      backgroundOffset: config.backgroundOffset,
      inputType,
      outputSize
    };
  }

  _buildDetectorRecord(config) {
    const interpreter = loadInterpreter(config.modelPath);
    let inputType = "float32";

    try {
      const inputTensor = interpreter.inputs[0];
      inputType = String(inputTensor?.type || "float32").toLowerCase();
    } catch (e) {
      console.warn("[Loader] Could not determine input type, defaulting to float32");
    }

    return {
      name: config.name,
      filePath: config.modelPath,
      interpreter,
      inputSize: config.inputSize,
      inputType
    };
  }

  async _loadClassifier(config) {
    await this._ensureEfficientNetExtracted(config);

    const primaryExists = await fileExists(config.modelPath);
    const fallbackExists = config.fallbackModelPath ? await fileExists(config.fallbackModelPath) : false;
    const resolvedPath = primaryExists ? config.modelPath : config.fallbackModelPath;

    if (!resolvedPath || (!primaryExists && !fallbackExists)) {
      throw new Error(`Missing classifier model for ${config.name}`);
    }

    return this._buildClassifierRecord(config, resolvedPath);
  }

  async _loadDetector() {
    const modelPath = path.resolve(DETECTOR_CONFIG.modelPath);
    console.log("[Loader] Loading detector from:", modelPath);

    try {
      await fsp.access(modelPath);
    } catch (error) {
      console.error(`[Loader] Detector model file NOT found at: ${modelPath}`);
      throw new Error(`Detector failed to load: File missing at ${modelPath}`);
    }

    try {
      const record = this._buildDetectorRecord(DETECTOR_CONFIG);
      console.log("[Loader] Detector loaded successfully");
      return record;
    } catch (error) {
      console.error(`[Loader] Detector failed to load: ${error.message}`);
      throw error;
    }
  }

  async _loadModels() {
    console.log("[Loader] Initializing models...");
    const classifierRecords = [];

    for (const config of CLASSIFIER_CONFIGS) {
      try {
        classifierRecords.push(await this._loadClassifier(config));
        console.log(`[Loader] Classifier '${config.name}' loaded.`);
      } catch (error) {
        console.error(`[Loader] Failed to load classifier '${config.name}': ${error.message}`);
        throw error;
      }
    }

    this._classifiers = classifierRecords;
    this._detector = null; // Detector is now handled by YOLO on-demand
    this._loadCount += 1;
    console.log("[Loader] All models initialized (YOLO detection enabled).");
  }
}

module.exports = new ModelLoader();