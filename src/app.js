const express = require("express");
const analyzeRouter = require("./routes/analyze");

const app = express();

app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    status: "ok",
    service: "material-detection"
  });
});

app.use("/analyze", analyzeRouter);

app.use((err, _req, res, _next) => {
  const message = err?.message || "Internal server error";
  const statusCode = /image|multipart|file|payload/i.test(message) ? 400 : 500;

  res.status(statusCode).json({
    error: message
  });
});

module.exports = { app };
