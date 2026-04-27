const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { app } = require("../src/app");
const modelLoader = require("../src/models/modelLoader");
const { ensureRealImages } = require("./support/fixtureGenerator");
const path = require("node:path");

const IMAGE_DIR = path.resolve(__dirname, "real_images");

let fixtures;

test.before(async () => {
  fixtures = await ensureRealImages(IMAGE_DIR);
  await modelLoader.initialize();
});

test("copper image resolves to copper", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.copperPath);

  assert.equal(response.status, 200);
  assert.equal(response.body.material, "copper");
  assert.ok(response.body.confidence > 0.4);
});

test("plastic image resolves to plastic", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.plasticPath);

  assert.equal(response.status, 200);
  assert.equal(response.body.material, "plastic");
});

test("iron image resolves to iron or metal", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.ironPath);

  assert.equal(response.status, 200);
  assert.match(response.body.material, /iron|metal/);
});

test("blurry image lowers confidence and is rejected", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.blurryPath);

  assert.equal(response.status, 200);
  assert.equal(response.body.accepted, false);
  assert.ok(response.body.confidence < 0.5);
  assert.equal(response.body.blur.isBlurry, true);
});

test("mixed image returns alternatives", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.mixedPath);

  assert.equal(response.status, 200);
  assert.ok(Array.isArray(response.body.alternatives));
  assert.ok(response.body.alternatives.length >= 1);
});

test("response contains the rebuilt pipeline payload", async () => {
  const response = await request(app).post("/analyze").attach("image", fixtures.copperPath);

  assert.equal(response.status, 200);
  assert.equal(typeof response.body.accepted, "boolean");
  assert.equal(typeof response.body.material, "string");
  assert.equal(typeof response.body.confidence, "number");
  assert.ok(Array.isArray(response.body.alternatives));
  assert.equal(typeof response.body.blur, "object");
  assert.equal(typeof response.body.blur.score, "number");
  assert.equal(typeof response.body.lighting, "object");
  assert.equal(typeof response.body.lighting.brightness, "number");
  assert.ok(Array.isArray(response.body.objects));
  assert.ok(Array.isArray(response.body.classifications));
  assert.equal(typeof response.body.features, "object");
  assert.equal(typeof response.body.features.brightness, "number");
  assert.equal(typeof response.body.features.variance, "number");
  assert.equal(typeof response.body.features.colorHistogram, "object");
  assert.equal(typeof response.body.weight, "number");
});

test("models load once as singletons", async () => {
  const before = modelLoader.getLoadCount();

  await request(app).post("/analyze").attach("image", fixtures.copperPath);
  await request(app).post("/analyze").attach("image", fixtures.plasticPath);

  const after = modelLoader.getLoadCount();
  assert.equal(before, after);
});
