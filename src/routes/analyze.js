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

router.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file?.buffer) {
      res.status(400).json({
        error: "Field 'image' with multipart/form-data is required."
      });
      return;
    }

    const imageBuffer = req.file.buffer;
    const [blurSummary, lightingSummary, detectionSummary, classificationSummary, featureSummary] = await Promise.all([
      blurService.analyze(imageBuffer),
      lightService.analyze(imageBuffer),
      detectionService.analyze(imageBuffer),
      classificationService.analyze(imageBuffer),
      featureService.extract(imageBuffer)
    ]);

    const fusion = fusionService.fuse({
      blurSummary,
      lightingSummary,
      detectionSummary,
      classificationSummary,
      featureSummary,
      embedding: classificationSummary?.embedding || []
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