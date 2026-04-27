const { app } = require("./app");
const modelLoader = require("./models/modelLoader");

const PORT = Number(process.env.PORT || 3023);

async function bootstrap() {
  await modelLoader.initialize();

  app.listen(PORT, () => {
    process.stdout.write(`material-detection service listening on port ${PORT}\n`);
  });
}

bootstrap().catch((error) => {
  console.error("Failed to start material detection service", error);
  process.exit(1);
});
