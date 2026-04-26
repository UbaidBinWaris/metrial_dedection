const fs = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const tf = require("@tensorflow/tfjs");
const tflite = require("@tensorflow/tfjs-tflite");
require("@tensorflow/tfjs-backend-wasm");
const sharp = require("sharp");

const IMAGE_SIZE = 224;

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  if (sum === 0) {
    return exps.map(() => 0);
  }
  return exps.map((value) => value / sum);
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

class ModelService {
  constructor() {
    this._models = [];
    this._initPromise = null;
    this._loadCount = 0;
    this._fallbackMode = false;
  }

  getLoadCount() {
    return this._loadCount;
  }

  async init() {
    if (this._initPromise) {
      return this._initPromise;
    }
    this._initPromise = this._loadModels();
    return this._initPromise;
  }

  _ensureEfficientNet() {
    const root = path.resolve(__dirname, "../../");
    const targetDir = path.join(root, "efficientnet-lite0");
    const targetModelFp32 = path.join(targetDir, "efficientnet-lite0-fp32.tflite");

    return fs
      .access(targetModelFp32)
      .catch(() => {
        const tarFile = path.join(root, "efficientnet-lite0.tar.gz");
        try {
          execFileSync("tar", ["-xzf", tarFile], { cwd: root, stdio: "ignore" });
        } catch (error) {
          // Keep startup resilient; service can still work with a single model.
        }
      })
      .then(() => ({ targetDir, targetModelFp32 }));
  }

  async _loadModelFromPath(name, modelPath) {
    const bytes = await fs.readFile(modelPath);
    const model = await tflite.loadTFLiteModel(new Uint8Array(bytes));
    this._models.push({ name, model, modelPath });
  }

  async _loadModels() {
    const root = path.resolve(__dirname, "../../");
    const mobileNetPath = path.join(root, "mobilenet_v2_1.0_224.tflite");

    await tf.setBackend("wasm");
    await tf.ready();

    const { targetModelFp32 } = await this._ensureEfficientNet();
    const efficientInt8Path = path.join(root, "efficientnet-lite0", "efficientnet-lite0-int8.tflite");

    try {
      await this._loadModelFromPath("mobilenet_v2", mobileNetPath);
    } catch (error) {
      throw new Error(`Failed to load MobileNet model at ${mobileNetPath}: ${error.message}`);
    }

    try {
      await this._loadModelFromPath("efficientnet_lite0", targetModelFp32);
    } catch (errorFp32) {
      try {
        await this._loadModelFromPath("efficientnet_lite0_int8", efficientInt8Path);
      } catch (errorInt8) {
        // Allow startup with one model if EfficientNet is unavailable.
      }
    }

    if (this._models.length === 0) {
      throw new Error("No TFLite model could be loaded.");
    }

    this._loadCount += 1;
  }

  async _preprocessImage(imageBuffer) {
    const { data } = await sharp(imageBuffer)
      .resize(IMAGE_SIZE, IMAGE_SIZE, { fit: "cover" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelCount = IMAGE_SIZE * IMAGE_SIZE;
    const normalized = new Float32Array(pixelCount * 3);

    let sumR = 0;
    let sumG = 0;
    let sumB = 0;

    for (let i = 0; i < pixelCount; i += 1) {
      const r = data[i * 3];
      const g = data[i * 3 + 1];
      const b = data[i * 3 + 2];

      normalized[i * 3] = r / 255;
      normalized[i * 3 + 1] = g / 255;
      normalized[i * 3 + 2] = b / 255;

      sumR += r;
      sumG += g;
      sumB += b;
    }

    return {
      input: normalized,
      meanRGB: {
        r: sumR / pixelCount / 255,
        g: sumG / pixelCount / 255,
        b: sumB / pixelCount / 255
      }
    };
  }

  async _runSingleModel(modelEntry, inputArray) {
    const inputTensor = tf.tensor4d(inputArray, [1, IMAGE_SIZE, IMAGE_SIZE, 3], "float32");

    try {
      const outputTensor = modelEntry.model.predict(inputTensor);
      const outputData = Array.from(await outputTensor.data());
      outputTensor.dispose();

      const probabilities = softmax(outputData);
      return {
        modelName: modelEntry.name,
        probabilities,
        outputLength: probabilities.length
      };
    } finally {
      inputTensor.dispose();
    }
  }

  _buildFallbackProbabilities(meanRGB) {
    const logits = [
      meanRGB.r * 1.1 + meanRGB.g * 0.4,
      meanRGB.b * 1.15,
      meanRGB.g,
      (meanRGB.r + meanRGB.g + meanRGB.b) / 3,
      0.5
    ];
    return softmax(logits);
  }

  _ensemble(modelOutputs, fallbackProbabilities) {
    if (modelOutputs.length === 0) {
      return fallbackProbabilities;
    }

    const minLength = Math.min(...modelOutputs.map((entry) => entry.probabilities.length));
    const merged = new Array(minLength).fill(0);

    for (const output of modelOutputs) {
      for (let i = 0; i < minLength; i += 1) {
        merged[i] += output.probabilities[i] / modelOutputs.length;
      }
    }

    const sum = merged.reduce((acc, value) => acc + value, 0);
    return sum > 0 ? merged.map((value) => value / sum) : fallbackProbabilities;
  }

  _topK(probabilities, k = 5) {
    return probabilities
      .map((value, index) => ({ index, confidence: value }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, k);
  }

  _computeInferenceQuality(probabilities) {
    const safe = probabilities.filter((value) => value > 0);
    const entropy = safe.reduce((acc, value) => acc - value * Math.log(value), 0);
    const normalizedEntropy = probabilities.length > 1 ? entropy / Math.log(probabilities.length) : 0;
    const top = Math.max(...probabilities);
    return clamp(top * (1 - normalizedEntropy) + 0.05, 0, 1);
  }

  async infer(imageBuffer) {
    await this.init();

    const preprocessed = await this._preprocessImage(imageBuffer);
    const fallbackProbabilities = this._buildFallbackProbabilities(preprocessed.meanRGB);
    const outputs = [];

    for (const modelEntry of this._models) {
      try {
        const result = await this._runSingleModel(modelEntry, preprocessed.input);
        outputs.push(result);
      } catch (error) {
        this._fallbackMode = true;
      }
    }

    const probabilities = this._ensemble(outputs, fallbackProbabilities);
    const topK = this._topK(probabilities, 5);

    return {
      probabilities,
      topK,
      inferenceQuality: this._computeInferenceQuality(probabilities),
      metadata: {
        loadedModels: this._models.map((entry) => entry.name),
        loadCount: this._loadCount,
        fallbackMode: this._fallbackMode || outputs.length === 0
      }
    };
  }
}

module.exports = new ModelService();
