#!/bin/bash

# Server configuration
SERVER="root@5.9.51.79"
REMOTE_PROJECT_DIR="/root/metrial_dedection"

echo "==============================================="
echo "Packaging Models Locally..."
echo "==============================================="

# Find all valid models and label files
find models -maxdepth 1 \( -name "*.tflite" -o -name "*.onnx" -o -name "*.pt" -o -name "*labels.txt" \) > files_to_copy.txt

# Add directories and root files if they exist
if [ -d "models/ssd_mobilenet_v2_saved_model" ]; then
    echo "models/ssd_mobilenet_v2_saved_model" >> files_to_copy.txt
fi
if [ -f "mobilenet_v2_1.0_224.tflite" ]; then
    echo "mobilenet_v2_1.0_224.tflite" >> files_to_copy.txt
fi

# Add the EfficientNet tar file
if [ -f "models/efficientnet-lite0.tar.gz" ]; then
    echo "models/efficientnet-lite0.tar.gz" >> files_to_copy.txt
fi

# Create a compressed archive locally
tar -czvf payload_models.tar.gz -T files_to_copy.txt

echo -e "\n==============================================="
echo "Uploading Archive to Server (You will be prompted for password)"
echo "==============================================="
scp payload_models.tar.gz $SERVER:$REMOTE_PROJECT_DIR/

echo -e "\n==============================================="
echo "Extracting on Server (You may be prompted for password again)"
echo "==============================================="
ssh $SERVER "cd $REMOTE_PROJECT_DIR && tar -xzvf payload_models.tar.gz && rm payload_models.tar.gz && if [ -f 'models/efficientnet-lite0.tar.gz' ]; then tar -xzvf models/efficientnet-lite0.tar.gz; fi"

# Clean up local temporary files
rm files_to_copy.txt payload_models.tar.gz

echo -e "\n==============================================="
echo "✅ All required models have been successfully copied!"
echo "You can now run 'npm start' on the server."
echo "==============================================="
