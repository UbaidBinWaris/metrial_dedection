# Server Deployment Guide

Follow these instructions to pull the latest code and start the material-detection-service on your server.

## 1. Pull the Latest Code

Navigate to the project directory on your server and pull the latest changes from GitHub:

```bash
cd /path/to/metrial_dedection
git pull origin main
```

## 2. Install Dependencies

Whenever you pull new code, ensure your Node.js dependencies are up to date. The project requires **Node.js v20 or higher**.

```bash
# Verify your node version
node -v

# Install dependencies
npm install
```

## 3. Verify the Models are Downloaded

Since we removed the heavy 16 GB datasets from Git tracking to allow the push, your ML models (e.g., MobileNet, EfficientNet, YOLO) must be present in the `models/` directory on your server. If they are not tracked via Git, make sure to transfer the `.tflite`, `.onnx`, or `.pt` model files into the `models/` directory manually (via `scp`, `rsync`, or downloading them directly on the server).

## 4. Start the Server

### Option A: Standard Startup (Testing/Foreground)
To start the server simply in the foreground (it will stop if you close the terminal):

```bash
npm start
```

### Option B: Production Startup using PM2 (Recommended)
For production environments, it's highly recommended to use a process manager like **PM2** so the server runs in the background, automatically restarts on crashes, and boots on system startup.

```bash
# Install PM2 globally if you haven't already
npm install -g pm2

# Start the application using PM2
pm2 start src/server.js --name "material-detection"

# Save the PM2 process list to restart automatically on server reboot
pm2 save
pm2 startup
```

## 5. Check Server Status & Logs

If you used PM2, you can view the logs and status with:

```bash
# View active processes
pm2 status

# View live console logs
pm2 logs material-detection
```

## 6. Test the API

Verify the server is running by sending a quick test request from the server terminal:

```bash
curl http://localhost:3000/
# Note: replace localhost:3000 with your actual host/port if different.
```
