#!/bin/bash

# Step 0: Run the checkpoint conversion script
echo "Running Python checkpoint conversion..."
python convert_checkpoint.py

# Define variables
CHECKPOINT_PATH="trained_model"
CONFIG_PATH="train/llama3_8_B.json"
OUTPUT_DIR="speculator-converted"
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Step 1: Check if config.json exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "✗ config.json not found at $CONFIG_PATH"
  echo "Please create a config.json that matches your model architecture."
  exit 1
fi

cp "$CONFIG_PATH" "$CHECKPOINT_PATH/config.json"

# Step 2: Convert with Speculator CLI
echo "Converting to Speculator format..."
speculators convert \
  --eagle3 \
  "$CHECKPOINT_PATH" \
  "$OUTPUT_DIR" \
  "$BASE_MODEL" \
  --norm-before-residual \
  --validate
