const express = require("express");
const multer = require("multer");
const modelService = require("./services/modelService");
const featureService = require("./services/featureService");
const heuristicService = require("./services/heuristicService");

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 8 * 1024 * 1024
  },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype || !file.mimetype.startsWith("image/")) {
      cb(new Error("Only image uploads are supported."));
      return;
    }
    cb(null, true);
  }
});

const app = express();

app.get("/health", (_req, res) => {
  res.status(200).json({
    status: "ok"
  });
});

app.post("/analyze", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file || !req.file.buffer) {
      res.status(400).json({
        error: "Field 'image' with multipart/form-data is required."
      });
      return;
    }

    const [modelInference, features] = await Promise.all([
      modelService.infer(req.file.buffer),
      featureService.extractFeatures(req.file.buffer)
    ]);

    const heuristic = heuristicService.combine(modelInference, features);

    res.status(200).json({
      material: heuristic.material,
      confidence: heuristic.confidence,
      alternatives: heuristic.alternatives,
      features,
      reasoning: heuristic.reasoning
    });
  } catch (error) {
    next(error);
  }
});

app.use((err, _req, res, _next) => {
  const message = err && err.message ? err.message : "Internal server error";
  const statusCode = /image|multipart|file/i.test(message) ? 400 : 500;

  res.status(statusCode).json({
    error: message
  });
});

module.exports = { app };
