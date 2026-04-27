const express = require("express");
const multer = require("multer");
const blurService = require("../pipeline/blurService");
const lightService = require("../pipeline/lightService");
const detectionService = require("../pipeline/detectionService");
const classificationService = require("../pipeline/classificationService");
const featureService = require("../pipeline/featureService");
const weightService = require("../pipeline/weightService");
const fusionService = require("../pipeline/fusionService");

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024
  },
  fileFilter: (_req, file, callback) => {
    if (!file.mimetype?.startsWith("image/")) {
      callback(new Error("Only image uploads are supported."));
      return;
    }

    callback(null, true);
  }
});

function withFallback(stageName, fallbackValue) {
  return async (work) => {
    try {
      return {
        value: await work(),
        warning: null
      };
    } catch (error) {
      const message = error?.message || "unknown_error";
      console.error(`[Analyze] ${stageName} failed: ${message}`);
      return {
        value: fallbackValue,
        warning: `${stageName}_failed:${message}`
      };
    }
  };
}

router.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file?.buffer) {
      res.status(400).json({
        error: "Field 'image' with multipart/form-data is required."
      });
      return;
    }

    const imageBuffer = req.file.buffer;
    const safeRun = {
      blur: withFallback("blur", { isBlurry: false, score: 0 }),
      lighting: withFallback("lighting", { brightness: 0.5, isTooDark: false, isTooBright: false, status: "unknown" }),
      detection: withFallback("detection", { objects: [], degraded: true, warning: "detection unavailable" }),
      classification: withFallback("classification", {
        labels: [{ classId: -1, label: "unknown_material", score: 0.34 }],
        perModel: [],
        certainty: 0,
        embedding: []
      }),
      features: withFallback("features", {
        brightness: 0.5,
        variance: 0,
        saturation: 0,
        metallicScore: 0,
        highlightRatio: 0,
        colorHistogram: { red: [], green: [], blue: [] },
        averageColor: { red: 0, green: 0, blue: 0 },
        dimensions: { width: 0, height: 0 }
      })
    };

    const [blurResult, lightingResult, detectionResult, classificationResult, featureResult] = await Promise.all([
      safeRun.blur(() => blurService.analyze(imageBuffer)),
      safeRun.lighting(() => lightService.analyze(imageBuffer)),
      safeRun.detection(() => detectionService.analyze(imageBuffer)),
      safeRun.classification(() => classificationService.analyze(imageBuffer)),
      safeRun.features(() => featureService.extract(imageBuffer))
    ]);

    const blurSummary = blurResult.value;
    const lightingSummary = lightingResult.value;
    const detectionSummary = detectionResult.value;
    const classificationSummary = classificationResult.value;
    const featureSummary = featureResult.value;
    const stageWarnings = [
      blurResult.warning,
      lightingResult.warning,
      detectionResult.warning,
      classificationResult.warning,
      featureResult.warning
    ].filter(Boolean);

    const fusion = fusionService.fuse({
      blurSummary,
      lightingSummary,
      detectionSummary,
      classificationSummary,
      featureSummary,
      embedding: classificationSummary?.embedding || [],
      stageWarnings
    });

    const weight = weightService.estimate(fusion.material, detectionSummary.objects, featureSummary);

    res.status(200).json({
      accepted: fusion.accepted,
      material: fusion.material,
      confidence: fusion.confidence,
      alternatives: fusion.alternatives,
      blur: blurSummary,
      lighting: lightingSummary,
      objects: detectionSummary.objects,
      classifications: classificationSummary.labels,
      features: featureSummary,
      warnings: fusion.warnings,
      weight,
      decision: fusion.decision
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;