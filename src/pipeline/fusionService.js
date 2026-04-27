const fs = require("node:fs");
const path = require("node:path");
const embeddingService = require("./embeddingService");

const allowedMap = {
  bottle: "plastic",
  cup: "plastic",
  bag: "plastic",
  can: "metal",
  laptop: "metal",
  "cell phone": "metal",
  phone: "metal",
  wire: "copper",
  mouse: "plastic",
  keyboard: "plastic"
};

let Wd = 0.6;
let Wf = 0.3;
let Wc = 0.1;

function loadWeights() {
  try {
    const weightsPath = path.join(__dirname, "../config/fusionWeights.json");
    if (fs.existsSync(weightsPath)) {
      const config = JSON.parse(fs.readFileSync(weightsPath, "utf-8"));
      if (config.Wd !== undefined) Wd = config.Wd;
      if (config.Wf !== undefined) Wf = config.Wf;
      if (config.Wc !== undefined) Wc = config.Wc;
      console.log(`[Fusion] Loaded weights: Wd=${Wd}, Wf=${Wf}, Wc=${Wc}`);
    }
  } catch (error) {
    console.warn(`[Fusion] Could not load fusionWeights.json, using default weights (${error?.message || "unknown_error"})`);
  }
}
loadWeights();

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function normalizeLabel(label) {
  return String(label || "").trim().toLowerCase().replaceAll(" ", "_");
}

function computeMetalScore(features) {
  const { metallicScore = 0, saturation = 0, brightness = 0, highlightRatio = 0 } = features;
  // Metal should be reflective + neutral; penalize oversaturated and very dark regions.
  return metallicScore * (1 - saturation) * (0.35 + (0.65 * highlightRatio)) * (0.4 + (0.6 * brightness));
}

function computePlasticScore(features) {
  const { metallicScore = 0, saturation = 0 } = features;
  const averageColor = features?.averageColor || {};
  const red = averageColor.red || 0;
  const green = averageColor.green || 0;
  const blue = averageColor.blue || 0;
  const warmBias = clamp(red - ((green + blue) / 2), 0, 1);
  // Plastic is colorful + non-metallic
  return saturation * (1 - metallicScore) * (1 - (0.75 * warmBias));
}

function computeRubberScore(features) {
  const { metallicScore = 0, brightness = 0, highlightRatio = 0, saturation = 0 } = features;
  // Rubber is dark, non-reflective, low texture variation
  const low_variance_factor = Math.max(0, 1 - highlightRatio);
  const lowColorFactor = Math.max(0, 1 - (saturation * 0.7));
  return (1 - brightness) * (1 - metallicScore) * low_variance_factor * lowColorFactor;
}

function computeCopperScore(features) {
  const averageColor = features?.averageColor || {};
  const red = averageColor.red || 0;
  const green = averageColor.green || 0;
  const blue = averageColor.blue || 0;
  const saturation = features?.saturation || 0;
  const warmth = clamp(red - blue, 0, 1);
  const greenBlueSpread = clamp(green - blue, 0, 1);
  const score = 1.5
    * warmth
    * (0.55 + (0.45 * saturation))
    * (0.7 + (0.3 * greenBlueSpread));
  return clamp(score, 0, 1);
}

function buildAlternatives(normalizedScores, finalMaterial) {
  const ranked = Object.entries(normalizedScores)
    .filter(([material]) => material !== finalMaterial)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 2)
    .map(([material, confidence]) => ({
      material,
      confidence: Number(confidence.toFixed(4))
    }));

  if (ranked.length > 0) {
    return ranked;
  }

  if (finalMaterial === "unknown") {
    return [];
  }

  return [{ material: "unknown", confidence: Number((1 - (normalizedScores[finalMaterial] || 0)).toFixed(4)) }];
}

function buildQualityWarnings(blurSummary, lightingSummary) {
  const warnings = [];
  if (blurSummary?.isBlurry) warnings.push("TOO_BLURRY");
  if (lightingSummary?.isTooDark) warnings.push("TOO_DARK");
  if (lightingSummary?.isTooBright) warnings.push("TOO_BRIGHT");
  return warnings;
}

function fuse({ blurSummary, lightingSummary, detectionSummary, classificationSummary, featureSummary, embedding, stageWarnings = [] }) {
  const detectionScores = {};
  const featureScores = {};
  const classScores = {};
  
  // STEP 1 - DETECTION (ONLY ALLOWED LABELS)
  const objects = detectionSummary?.objects || [];
  let hasDetection = false;
  
  for (const obj of objects) {
    const label = normalizeLabel(obj.label);
    
    let matchedMaterial = null;
    for (const [key, val] of Object.entries(allowedMap)) {
      if (label.includes(key)) {
        matchedMaterial = val;
        break;
      }
    }
    
    // Ignore garbage predictions like person, bird, etc.
    if (matchedMaterial) {
      // Use core object score relative to Detection Weight
      detectionScores[matchedMaterial] = (detectionScores[matchedMaterial] || 0) + (obj.score * Wd);
      hasDetection = true;
    }
  }

  // STEP 2 - RAW FEATURE SCORES
  const features = featureSummary || {};
  const rawMetal = computeMetalScore(features);
  const rawPlastic = computePlasticScore(features);
  const rawRubber = computeRubberScore(features);
  const rawCopper = computeCopperScore(features);
  const glassScore = Math.max(0, (1 - (features.saturation || 0)) * (features.brightness || 0) - (features.metallicScore || 0));

  console.log(`[Fusion] Raw feature scores: { metal: ${rawMetal.toFixed(4)}, plastic: ${rawPlastic.toFixed(4)}, rubber: ${rawRubber.toFixed(4)}, copper: ${rawCopper.toFixed(4)} }`);

  // Assign features multiplied by Wf
  featureScores["metal"] = rawMetal * Wf;
  featureScores["plastic"] = rawPlastic * Wf;
  featureScores["rubber"] = rawRubber * Wf;
  featureScores["copper"] = rawCopper * Wf;
  if (glassScore > 0) featureScores["glass"] = glassScore * Wf;

  // STEP 3 - FEATURE PENALTY (METAL REQUIRES EVIDENCE)
  if (!hasDetection) {
    featureScores["metal"] *= 0.75;
  }

  // STEP 4 - CLASSIFICATION (Low Impact, Safe Labels Only)
  if (classificationSummary?.certainty > 0.05) {
    const labels = classificationSummary?.labels || [];
    for (const item of labels) {
      if (item.score > 0.05) {
        const label = normalizeLabel(item.label);
        let matchedMaterial = null;
        for (const [key, val] of Object.entries(allowedMap)) {
          if (label.includes(key)) {
            matchedMaterial = val;
            break;
          }
        }
        if (matchedMaterial) {
          classScores[matchedMaterial] = (classScores[matchedMaterial] || 0) + (item.score * Wc);
        }
      }
    }
  }

  // AGGREGATE ADJUSTED SCORES
  let adjustedScores = {};
  const allKeys = new Set([...Object.keys(detectionScores), ...Object.keys(featureScores), ...Object.keys(classScores)]);

  for (const key of allKeys) {
    adjustedScores[key] = (detectionScores[key] || 0) + (featureScores[key] || 0) + (classScores[key] || 0);
  }

  const formatScores = (scores) => {
    const entries = Object.entries(scores).map(([k, v]) => `${k}: ${Number(v.toFixed(4))}`);
    return entries.length > 0 ? `{ ${entries.join(", ")} }` : `{}`;
  };

  console.log(`[Fusion] Adjusted scores: ${formatScores(adjustedScores)}`);

  // STEP 5 - NORMALIZE SCORES
  let total = 0;
  for (const score of Object.values(adjustedScores)) {
    total += score;
  }

  const normalizedScores = {};
  let bestMaterial = "unknown";
  let highestNormalizedScore = 0;

  if (total > 0) {
    for (const [mat, score] of Object.entries(adjustedScores)) {
      const normScore = score / total;
      normalizedScores[mat] = normScore;
      if (normScore > highestNormalizedScore) {
        highestNormalizedScore = normScore;
        bestMaterial = mat;
      }
    }
  }

  console.log(`[Fusion] Final normalized scores: ${formatScores(normalizedScores)}`);

  // STEP 6 - EMBEDDING-FIRST DECISION & HYBRID BOOST
  let embedMatch = { material: "unknown", confidence: 0 };
  if (embedding && embedding.length > 0) {
    embedMatch = embeddingService.findSimilar(embedding, 3);
  }

  let finalMaterial;
  let finalConfidence;
  let decisionSource;
  const strongestDetection = Math.max(...Object.values(detectionScores), 0);
  const embeddingSupportFromFusion = embedMatch.material && embedMatch.material !== "unknown"
    ? (normalizedScores[embedMatch.material] || 0)
    : 0;
  const embeddingIsReliable = embedMatch.confidence >= 0.72
    && (strongestDetection >= 0.18 || embeddingSupportFromFusion >= 0.2);

  if (embedMatch.material !== "unknown" && embeddingIsReliable) {
    console.log(`[Decision] Using EMBEDDING (confidence: ${embedMatch.confidence.toFixed(2)})`);
    finalMaterial = embedMatch.material;
    finalConfidence = embedMatch.confidence;
    decisionSource = embedMatch.source;
  } else if (embedMatch.confidence >= 0.55 && embedMatch.material !== "unknown" && embeddingSupportFromFusion >= 0.12) {
    const fusionScore = normalizedScores[embedMatch.material] || 0;
    const hybridScore = (embedMatch.confidence * 0.7) + (fusionScore * 0.3);
    
    console.log(`[Decision] Hybrid mode activated. Embd: ${embedMatch.confidence.toFixed(2)}, Fusion: ${fusionScore.toFixed(2)} -> Hybrid: ${hybridScore.toFixed(2)}`);
    finalMaterial = embedMatch.material;
    finalConfidence = hybridScore;
    decisionSource = "hybrid_fusion";
  } else if (highestNormalizedScore >= 0.4 && bestMaterial !== "unknown") {
    console.log(`[Decision] Falling back to FUSION (confidence: ${highestNormalizedScore.toFixed(2)})`);
    finalMaterial = bestMaterial;
    finalConfidence = highestNormalizedScore;
    decisionSource = "weighted_fusion";
  } else {
    console.log(`[Decision] Both engines weak. Returning unknown.`);
    finalMaterial = "unknown";
    finalConfidence = 0.1;
    decisionSource = "uncertain";
  }

  console.log(`[Decision] Selected: ${finalMaterial}`);

  const qualityWarnings = buildQualityWarnings(blurSummary, lightingSummary);
  const qualityPenalty = (blurSummary?.isBlurry ? 0.25 : 0)
    + (lightingSummary?.isTooDark ? 0.18 : 0)
    + (lightingSummary?.isTooBright ? 0.1 : 0);
  const confidenceAfterQuality = clamp(finalConfidence * (1 - qualityPenalty), 0.05, 1);
  const hardReject = Boolean(blurSummary?.isBlurry) && Boolean(lightingSummary?.isTooDark);
  const accepted = !(blurSummary?.isBlurry) && !hardReject && confidenceAfterQuality >= 0.35;

  const warnings = [
    ...qualityWarnings,
    ...(detectionSummary?.warning ? [detectionSummary.warning] : []),
    ...stageWarnings
  ];

  return {
    accepted,
    material: finalMaterial,
    confidence: Number(confidenceAfterQuality.toFixed(4)),
    alternatives: buildAlternatives(normalizedScores, finalMaterial),
    warnings,
    decision: { 
      source: decisionSource,
      reasons: decisionSource === "uncertain"
        ? ["low-signal", ...qualityWarnings.map((item) => `quality:${item.toLowerCase()}`)]
        : qualityWarnings.map((item) => `quality:${item.toLowerCase()}`)
    }
  };
}

/**
 * OPTIONAL: Dataset-Based Weight Tuning (CPU Only)
 */
function tuneWeightsGridSearch(datasetItems) {
  console.log("Starting Grid Search for Dataset Tuning...");
  let bestAccuracy = 0;
  let bestWeights = { Wd: 0.6, Wf: 0.3, Wc: 0.1 };

  for (let WdTest = 0.4; WdTest <= 0.7; WdTest += 0.1) {
    for (let WfTest = 0.2; WfTest <= 0.5; WfTest += 0.1) {
      for (let WcTest = 0.05; WcTest <= 0.2; WcTest += 0.05) {
        let correctCount = 0;
        
        for (const item of datasetItems) {
          let scores = { metal: 0, plastic: 0, glass: 0, rubber: 0 };
          
          if (item.objScore > 0.4) scores[item.trueMaterial] += (item.objScore * WdTest);
          
          const rawMetal = item.metallicScore * (1 - item.saturation) * (1 - (item.brightness || 0));
          const rawPlastic = item.saturation * (1 - item.metallicScore);
          const rawRubber = (1 - (item.brightness || 0)) * (1 - item.metallicScore) * (1 - (item.highlightRatio || 0));

          let fMetal = rawMetal * WfTest;
          if (item.objScore < 0.4) fMetal *= 0.5; // Penalty

          scores["metal"] = (scores["metal"] || 0) + fMetal;
          scores["plastic"] = (scores["plastic"] || 0) + rawPlastic * WfTest;
          scores["rubber"] = (scores["rubber"] || 0) + rawRubber * WfTest;
          
          let bestMaterial = "unknown";
          let maxScore = 0;
          for (const [mat, score] of Object.entries(scores)) {
            if (score > maxScore) {
              maxScore = score;
              bestMaterial = mat;
            }
          }
          
          if (bestMaterial === item.trueMaterial) correctCount++;
        }
        
        const accuracy = correctCount / datasetItems.length;
        if (accuracy > bestAccuracy) {
          bestAccuracy = accuracy;
          bestWeights = { Wd: WdTest, Wf: WfTest, Wc: WcTest };
        }
      }
    }
  }
  
  console.log(`Tuning Complete! Best Weights: Wd=${bestWeights.Wd.toFixed(2)}, Wf=${bestWeights.Wf.toFixed(2)}, Wc=${bestWeights.Wc.toFixed(2)} (Acc: ${(bestAccuracy*100).toFixed(1)}%)`);
  
  try {
    const weightsPath = path.join(__dirname, "../config/fusionWeights.json");
    fs.writeFileSync(weightsPath, JSON.stringify(bestWeights, null, 2));
    console.log(`[Fusion] Saved new weights to ${weightsPath}`);
    // Live update variables
    Wd = bestWeights.Wd;
    Wf = bestWeights.Wf;
    Wc = bestWeights.Wc;
  } catch(e) {
    console.error("[Fusion] Failed to save weights", e);
  }

  return bestWeights;
}

module.exports = {
  fuse,
  tuneWeightsGridSearch
};