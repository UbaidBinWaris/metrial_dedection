const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { app } = require("../src/app");
const modelService = require("../src/services/modelService");
const { ensureRealImages } = require("../src/utils/fixtureGenerator");

const IMAGE_DIR = path.resolve(__dirname, "./real_images");

let fixtures;

test.before(async () => {
  fixtures = await ensureRealImages(IMAGE_DIR);
  await modelService.init();
});

test("Copper image -> predicted copper", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.copperPath);

  assert.equal(response.status, 200);
  assert.equal(response.body.material, "copper");
  assert.equal(typeof response.body.confidence, "number");
});

test("Iron image -> predicted iron", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.ironPath);

  assert.equal(response.status, 200);
  assert.equal(response.body.material, "iron");
});

test("Blurry image -> low confidence", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.blurryPath);

  assert.equal(response.status, 200);
  assert.ok(response.body.confidence < 0.65);
});

test("Mixed material -> top 2 alternatives returned", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.mixedPath);

  assert.equal(response.status, 200);
  assert.ok(Array.isArray(response.body.alternatives));
  assert.ok(response.body.alternatives.length >= 2);
});

test("Output format matches spec", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.plasticPath);

  assert.equal(response.status, 200);
  assert.equal(typeof response.body.material, "string");
  assert.equal(typeof response.body.confidence, "number");
  assert.equal(typeof response.body.reasoning, "string");
  assert.equal(typeof response.body.features, "object");
  assert.equal(typeof response.body.features.brightness, "number");
  assert.equal(typeof response.body.features.variance, "number");
  assert.equal(typeof response.body.features.colorDistribution, "object");
});

test("Model loads once (singleton)", async () => {
  const before = modelService.getLoadCount();

  await request(app).post("/analyze").attach("image", fixtures.copperPath);
  await request(app).post("/analyze").attach("image", fixtures.plasticPath);

  const after = modelService.getLoadCount();
  assert.equal(before, after);
});
