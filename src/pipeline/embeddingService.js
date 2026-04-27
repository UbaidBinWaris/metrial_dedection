const fs = require('node:fs');
const path = require('node:path');

let embeddingsDB = [];
let classCentroids = {};
let dbLoaded = false;

const MIN_TOP_SIMILARITY = 0.7;
const MIN_VECTOR_STDDEV = 0.00001;

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function standardDeviation(values) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => {
    const diff = value - mean;
    return sum + (diff * diff);
  }, 0) / values.length;
  return Math.sqrt(variance);
}

function computeCentroids() {
  const sums = {};
  const counts = {};
  
  for (const entry of embeddingsDB) {
    const label = entry.label;
    if (!sums[label]) {
      sums[label] = new Array(entry.embedding.length).fill(0);
      counts[label] = 0;
    }
    for (let i = 0; i < entry.embedding.length; i++) {
      sums[label][i] += entry.embedding[i];
    }
    counts[label]++;
  }

  for (const label of Object.keys(sums)) {
    classCentroids[label] = sums[label].map(val => val / counts[label]);
  }
}

function loadEmbeddings() {
  if (dbLoaded) return;
  const dbPath = path.join(__dirname, '../config/embeddings.json');
  if (fs.existsSync(dbPath)) {
    try {
      embeddingsDB = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
      computeCentroids();
      console.log(`[Embedding] Loaded ${embeddingsDB.length} samples and built centroids for ${Object.keys(classCentroids).length} classes.`);
    } catch (e) {
      console.error(`[Embedding] Failed to parse embeddings.json`, e);
    }
  }
  dbLoaded = true;
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function magnitude(a) {
  let sum = 0;
  for (const value of a) sum += value * value;
  return Math.sqrt(sum);
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dotProduct(a, b) / (magA * magB);
}

function findSimilar(targetEmbedding, k = 5) {
  loadEmbeddings();
  
  if (embeddingsDB.length === 0) {
    return { material: "unknown", confidence: 0, source: "embedding_similarity_failed" };
  }

  // Calculate similarity for all
  const similarities = embeddingsDB.map(entry => {
    return {
      label: entry.label,
      filename: entry.filename,
      sim: cosineSimilarity(targetEmbedding, entry.embedding)
    };
  });

  // Sort descending
  similarities.sort((a, b) => b.sim - a.sim);

  const top1 = similarities[0] || { sim: 0 };
  const top2 = similarities[1] || { sim: 0 };

  // 1. OUTLIER PROTECTION
  if (top1.sim < MIN_TOP_SIMILARITY) {
    console.log(`[Embedding] Outlier detected. Top similarity ${top1.sim.toFixed(2)} < ${MIN_TOP_SIMILARITY}. Returning unknown.`);
    return { material: "unknown", confidence: 0.1, source: "embedding_outlier" };
  }

  const embeddingStdDev = standardDeviation(targetEmbedding);
  if (embeddingStdDev < MIN_VECTOR_STDDEV) {
    console.log(`[Embedding] Low-variance embedding (${embeddingStdDev.toExponential(2)}). Returning unknown.`);
    return { material: "unknown", confidence: 0.1, source: "embedding_low_variance" };
  }

  // Top K
  const topK = similarities.slice(0, k);
  
  // 2. WEIGHTED KNN VOTING
  const materialScores = {};
  for (const match of topK) {
    materialScores[match.label] = (materialScores[match.label] || 0) + match.sim;
  }
  
  let bestLabel = "unknown";
  let maxScore = 0;
  for (const [label, score] of Object.entries(materialScores)) {
    if (score > maxScore) {
      maxScore = score;
      bestLabel = label;
    }
  }

  // 3. CONFIDENCE SMOOTHING
  const winningMatches = topK.filter(m => m.label === bestLabel);
  const knnScore = winningMatches.length > 0 
    ? winningMatches.reduce((acc, m) => acc + m.sim, 0) / winningMatches.length 
    : 0;

  // 4. CLASS CENTROIDS
  const centroid = classCentroids[bestLabel];
  const centroidScore = centroid ? cosineSimilarity(targetEmbedding, centroid) : 0;

  // 5. HYBRID CONFIDENCE
  const baseConfidence = (0.7 * knnScore) + (0.3 * centroidScore);
  const topMargin = Math.max(0, top1.sim - top2.sim);
  const marginFactor = clamp(topMargin / 0.04, 0.25, 1);
  const supportFactor = clamp(embeddingsDB.length / 30, 0.25, 1);
  const varianceFactor = clamp((embeddingStdDev - MIN_VECTOR_STDDEV) / 0.00015, 0.3, 1);
  const finalConfidence = clamp(baseConfidence * marginFactor * supportFactor * varianceFactor, 0, 1);

  // 6. DEBUG LOGS
  console.log(`[Embedding] Top ${k} neighbors:`);
  topK.forEach(m => console.log(`  ${m.label}: ${m.sim.toFixed(4)}`));
  console.log(`[Embedding] Final vote: ${bestLabel} (KNN: ${knnScore.toFixed(4)}, Centroid: ${centroidScore.toFixed(4)}, Margin: ${topMargin.toFixed(4)}, Support: ${supportFactor.toFixed(2)}, Var: ${embeddingStdDev.toExponential(2)} -> Conf: ${finalConfidence.toFixed(4)})`);
  
  // Normalize dataset label if it matches the folder name
  if (bestLabel === "metel_iron" || bestLabel === "metal_iron") {
    bestLabel = "metal";
  }
  
  return {
    material: bestLabel,
    confidence: Number(finalConfidence.toFixed(4)),
    source: "embedding_similarity"
  };
}

module.exports = {
  findSimilar,
  cosineSimilarity
};
