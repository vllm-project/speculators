#!/bin/bash
# setup.sh - Install dependencies for evaluating speculator models

set -e

echo "Installing dependencies..."

pip install -r requirements.txt

echo "Setup complete!"
