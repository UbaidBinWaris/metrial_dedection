const { app } = require("./app");
const modelService = require("./services/modelService");

const PORT = Number(process.env.PORT || 3000);

async function bootstrap() {
  await modelService.init();

  app.listen(PORT, () => {
    console.log(`Material detection service running on port ${PORT}`);
    console.log(`Model load count: ${modelService.getLoadCount()}`);
  });
}

bootstrap().catch((error) => {
  console.error("Failed to start material detection service", error);
  process.exit(1);
});
