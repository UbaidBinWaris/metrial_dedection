const fs = require("node:fs");
const path = require("node:path");
const sharp = require("sharp");
const modelLoader = require("../models/modelLoader");

let imagenetLabels = null;

function loadLabels() {
  if (imagenetLabels) return imagenetLabels;
  const labelsPath = path.join(__dirname, "../../models/imagenet_labels.txt");
  try {
    const content = fs.readFileSync(labelsPath, "utf8");
    imagenetLabels = content.split("\n").map(line => line.trim()).filter(line => line);
    return imagenetLabels;
  } catch (error) {
    console.error(`Failed to load ImageNet labels: ${error.message}`);
    return [];
  }
}

function softmax(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return [];
  }

  const maxValue = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maxValue));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(sum) || sum <= 0) {
    const uniform = 1 / values.length;
    return values.map(() => uniform);
  }

  return exponentials.map((value) => value / sum);
}

function alignProbabilities(probabilities, backgroundOffset) {
  return backgroundOffset > 0 ? probabilities.slice(backgroundOffset) : probabilities;
}

function topK(probabilities, limit) {
  if (!Array.isArray(probabilities) || probabilities.length === 0) {
    return [];
  }

  const labels = loadLabels();

  return probabilities
    .map((score, index) => ({
      classId: index,
      label: labels[index] || `class_${index}`,
      score: Number.isFinite(score) ? score : 0
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map((entry) => ({
      classId: entry.classId,
      label: entry.label,
      score: Number(entry.score.toFixed(4))
    }));
}

async function preprocess(imageBuffer, modelConfig) {
  const { data, info } = await sharp(imageBuffer)
    .rotate()
    .resize(modelConfig.inputSize, modelConfig.inputSize, { fit: "cover" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixelCount = info.width * info.height;

  if (modelConfig.inputType.includes("uint8")) {
    return new Uint8Array(data);
  }

  const normalized = new Float32Array(pixelCount * 3);
  for (let index = 0; index < pixelCount * 3; index += 1) {
    const value = data[index] / 255;
    normalized[index] = modelConfig.normalization === "signed" ? (value * 2) - 1 : value;
  }

  return normalized;
}

function runModel(modelConfig, input) {
  const output = new Float32Array(modelConfig.outputSize);
  modelConfig.interpreter.inputs[0].copyFrom(input);
  modelConfig.interpreter.invoke();
  modelConfig.interpreter.outputs[0].copyTo(output);

  const probabilities = alignProbabilities(softmax(Array.from(output)), modelConfig.backgroundOffset);

  return {
    model: modelConfig.name,
    topLabels: topK(probabilities, 3),
    probabilities
  };
}

function ensemble(modelOutputs) {
  if (!Array.isArray(modelOutputs) || modelOutputs.length === 0) {
    return [];
  }

  const probabilityLength = Math.min(...modelOutputs.map((output) => output.probabilities.length));
  if (!Number.isFinite(probabilityLength) || probabilityLength <= 0) {
    return [];
  }

  const merged = new Array(probabilityLength).fill(0);

  for (const output of modelOutputs) {
    for (let index = 0; index < probabilityLength; index += 1) {
      merged[index] += output.probabilities[index] / modelOutputs.length;
    }
  }

  return merged;
}

async function analyze(imageBuffer) {
  await modelLoader.initialize();

  const classifierModels = modelLoader.getClassifierModels();
  const modelOutputs = [];

  for (const modelConfig of classifierModels) {
    const input = await preprocess(imageBuffer, modelConfig);
    modelOutputs.push(runModel(modelConfig, input));
  }

  const combinedProbabilities = ensemble(modelOutputs);
  const labels = topK(combinedProbabilities, 3);

  const safeLabels = labels.length > 0
    ? labels
    : [
        { classId: -1, label: "unknown_material", score: 0.34 },
        { classId: -2, label: "unknown_object", score: 0.33 },
        { classId: -3, label: "unknown_surface", score: 0.33 }
      ];
  const certainty = safeLabels[0]?.score || 0;

  console.log(`[Classification] Top label: ${safeLabels[0]?.label} (${safeLabels[0]?.score})`);

  return {
    labels: safeLabels,
    perModel: modelOutputs.map((output) => ({
      model: output.model,
      topLabels: output.topLabels
    })),
    certainty: Number(certainty.toFixed(4)),
    embedding: combinedProbabilities
  };
}

module.exports = {
  analyze
};