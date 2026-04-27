const DENSITY_KG_PER_M3 = {
  copper: 8960,
  plastic: 950,
  iron: 7870,
  aluminum: 2700,
  mixed: 1800,
  metal: 7800,
  glass: 2500
};

const OBJECT_BASE_VOLUME_M3 = {
  bottle: 0.001,
  cup: 0.00035,
  laptop: 0.0012,
  keyboard: 0.0008,
  cell_phone: 0.00008,
  mouse: 0.00009,
  bowl: 0.0004,
  book: 0.0007,
  wire: 0.00005,
  electronics: 0.0003
};

const MATERIAL_BASE_VOLUME_M3 = {
  copper: 0.00012,
  plastic: 0.00045,
  iron: 0.00015,
  aluminum: 0.0003,
  mixed: 0.00025,
  metal: 0.00018,
  glass: 0.00035
};

function normalizeLabel(label) {
  return String(label || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function estimateCoverage(objects) {
  if (!Array.isArray(objects) || objects.length === 0) {
    return 0.25;
  }

  const total = objects.reduce((coverage, objectEntry) => {
    const box = objectEntry.box;
    if (!box) {
      return coverage;
    }

    const width = Math.max(0, box.right - box.left);
    const height = Math.max(0, box.bottom - box.top);
    return coverage + (width * height);
  }, 0);

  return clamp(total, 0.08, 0.9);
}

function estimate(material, objects, featureSummary) {
  const normalizedObjects = Array.isArray(objects) ? objects : [];
  const firstObject = normalizedObjects[0] || null;
  const normalizedObjectLabel = normalizeLabel(firstObject?.label);
  const density = DENSITY_KG_PER_M3[material] || DENSITY_KG_PER_M3.mixed;
  const coverage = estimateCoverage(normalizedObjects);
  const brightnessFactor = 0.8 + ((featureSummary?.brightness || 0.5) * 0.4);
  const varianceFactor = 0.85 + Math.min(0.35, (featureSummary?.variance || 0.1) * 2);
  const objectVolume = OBJECT_BASE_VOLUME_M3[normalizedObjectLabel] || MATERIAL_BASE_VOLUME_M3[material] || MATERIAL_BASE_VOLUME_M3.mixed;
  const estimatedVolume = objectVolume * coverage * brightnessFactor * varianceFactor;
  const estimatedWeight = density * estimatedVolume;

  return Number(Math.max(0.01, estimatedWeight).toFixed(3));
}

module.exports = {
  estimate
};