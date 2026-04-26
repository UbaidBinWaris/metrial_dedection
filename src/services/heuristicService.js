const { MATERIALS } = require("../config/materialProfiles");

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function inRangeScore(value, min, max) {
  if (value >= min && value <= max) {
    return 1;
  }
  if (value < min) {
    return clamp(1 - (min - value) * 3, 0, 1);
  }
  return clamp(1 - (value - max) * 3, 0, 1);
}

function normalizeScores(rawScores) {
  const sum = rawScores.reduce((acc, score) => acc + score, 0);
  if (sum <= 0) {
    const uniform = 1 / rawScores.length;
    return rawScores.map((_, idx) => ({ material: MATERIALS[idx], confidence: uniform }));
  }

  return rawScores
    .map((score, idx) => ({ material: MATERIALS[idx], confidence: score / sum }))
    .sort((a, b) => b.confidence - a.confidence);
}

class HeuristicService {
  combine(modelInference, features) {
    const { brightness, variance, colorDistribution } = features;
    const meanRGB = colorDistribution.meanRGB;
    const saturation = colorDistribution.saturation;
    const warmBias = meanRGB.r - meanRGB.b;
    const neutralBias = 1 - Math.abs(meanRGB.r - meanRGB.g) - Math.abs(meanRGB.g - meanRGB.b);
    const modelQuality = modelInference.inferenceQuality;

    const rawScores = [
      // copper
      0.42 * clamp((warmBias + 0.15) / 0.45) +
        0.22 * saturation +
        0.2 * inRangeScore(brightness, 0.25, 0.82) +
        0.16 * modelQuality,

      // iron
      0.35 * (1 - saturation) +
        0.25 * inRangeScore(brightness, 0.18, 0.62) +
        0.24 * inRangeScore(variance, 0.07, 0.28) +
        0.16 * modelQuality,

      // plastic
      0.32 * saturation +
        0.24 * inRangeScore(variance, 0.02, 0.2) +
        0.24 * inRangeScore(brightness, 0.2, 0.88) +
        0.2 * modelQuality,

      // aluminum
      0.33 * (1 - saturation) +
        0.32 * inRangeScore(brightness, 0.44, 0.95) +
        0.15 * clamp((neutralBias + 0.1) / 1.1) +
        0.2 * modelQuality,

      // glass
      0.32 * inRangeScore(brightness, 0.35, 0.95) +
        0.28 * (1 - saturation) +
        0.2 * inRangeScore(variance, 0.01, 0.16) +
        0.2 * modelQuality,

      // wood
      0.34 * clamp((warmBias + 0.08) / 0.35) +
        0.28 * inRangeScore(brightness, 0.14, 0.7) +
        0.2 * inRangeScore(variance, 0.05, 0.32) +
        0.18 * modelQuality
    ].map((score) => clamp(score, 0.001, 1));

    const ranked = normalizeScores(rawScores);
    const primary = ranked[0];
    const secondary = ranked[1];

    const blurry = variance < 0.05;
    const mixed = secondary && primary.confidence - secondary.confidence < 0.08;

    let finalConfidence = primary.confidence;
    if (blurry) {
      finalConfidence *= 0.55;
    }
    if (mixed) {
      finalConfidence *= 0.82;
    }

    finalConfidence = clamp(finalConfidence * (0.7 + modelQuality * 0.3));

    const alternatives = ranked
      .slice(1, mixed ? 3 : 2)
      .filter((entry) => entry.confidence > 0.08)
      .map((entry) => ({
        material: entry.material,
        confidence: Number(entry.confidence.toFixed(4))
      }));

    return {
      material: primary.material,
      confidence: Number(finalConfidence.toFixed(4)),
      alternatives,
      reasoning: this._buildReasoning(primary.material, {
        brightness,
        variance,
        saturation,
        warmBias,
        blurry,
        mixed
      })
    };
  }

  _buildReasoning(material, metrics) {
    const tone =
      metrics.warmBias > 0.08
        ? "warm tones"
        : metrics.warmBias < -0.05
          ? "cool tones"
          : "neutral tones";

    const reflectivity = metrics.brightness > 0.62 ? "high reflectivity" : "moderate reflectivity";
    const texture = metrics.variance > 0.14 ? "visible texture variance" : "smoother surface texture";

    let sentence = `${reflectivity} and ${tone} with ${texture} suggest ${material}.`;

    if (metrics.blurry) {
      sentence += " Low variance indicates blur, so confidence was reduced.";
    }

    if (metrics.mixed) {
      sentence += " The top material scores are close, indicating a potential mixed-material scene.";
    }

    return sentence;
  }
}

module.exports = new HeuristicService();
