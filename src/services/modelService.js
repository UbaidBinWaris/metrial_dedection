const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { Interpreter } = require("node-tflite");
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
    if (this._initPromise !== null) {
      return this._initPromise;
    }
    this._initPromise = this._loadModels();
    return this._initPromise;
  }

  _ensureEfficientNet() {
    const root = path.resolve(__dirname, "../../");
    const targetModelFp32 = path.join(root, "efficientnet-lite0", "efficientnet-lite0-fp32.tflite");

    return fsp
      .access(targetModelFp32)
      .catch(() => {
        const tarFile = path.join(root, "efficientnet-lite0.tar.gz");
        try {
          execFileSync("tar", ["-xzf", tarFile], { cwd: root, stdio: "ignore" });
        } catch (error_) {
          console.warn("EfficientNet archive extraction failed; continuing with available models.", error_.message);
        }
      })
      .then(() => targetModelFp32);
  }

  _detectOutputLength(interpreter) {
    const output = interpreter.outputs?.[0] ?? null;
    const dims = output?.dims;
    if (Array.isArray(dims) && dims.length > 0) {
      const computed = dims.reduce((acc, value) => acc * value, 1);
      if (Number.isFinite(computed) && computed > 0) {
        return computed;
      }
    }
    return 1001;
  }

  _loadModelFromPath(name, modelPath) {
    const bytes = fs.readFileSync(modelPath);
    const interpreter = new Interpreter(bytes);
    interpreter.allocateTensors();

    this._models.push({
      name,
      interpreter,
      modelPath,
      outputLength: this._detectOutputLength(interpreter)
    });
  }

  async _loadModels() {
    const root = path.resolve(__dirname, "../../");
    const mobileNetPath = path.join(root, "mobilenet_v2_1.0_224.tflite");

    const targetModelFp32 = await this._ensureEfficientNet();
    const efficientInt8Path = path.join(root, "efficientnet-lite0", "efficientnet-lite0-int8.tflite");

    try {
      this._loadModelFromPath("mobilenet_v2", mobileNetPath);
    } catch (error) {
      throw new Error(`Failed to load MobileNet model at ${mobileNetPath}: ${error.message}`);
    }

    try {
      this._loadModelFromPath("efficientnet_lite0", targetModelFp32);
    } catch (error_) {
      try {
        this._loadModelFromPath("efficientnet_lite0_int8", efficientInt8Path);
      } catch (error_) {
        console.warn("EfficientNet model unavailable; running with MobileNet only.", error_.message);
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

      normalized[i * 3] = (r / 127.5) - 1;
      normalized[i * 3 + 1] = (g / 127.5) - 1;
      normalized[i * 3 + 2] = (b / 127.5) - 1;

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

  _runSingleModel(modelEntry, inputArray) {
    const input = new Float32Array(inputArray);
    const output = new Float32Array(modelEntry.outputLength);

    modelEntry.interpreter.inputs[0].copyFrom(input);
    modelEntry.interpreter.invoke();
    modelEntry.interpreter.outputs[0].copyTo(output);

    const probabilities = softmax(Array.from(output));
    return {
      modelName: modelEntry.name,
      probabilities,
      outputLength: probabilities.length
    };
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
        const result = this._runSingleModel(modelEntry, preprocessed.input);
        outputs.push(result);
      } catch (error_) {
        console.warn(`Inference failed for ${modelEntry.name}; skipping this model.`, error_.message);
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
