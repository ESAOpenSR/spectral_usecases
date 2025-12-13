#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
ZIP_URL="https://huggingface.co/datasets/simon-donike/viz/resolve/main/data_usecases.zip"
ZIP_NAME="data_usecases.zip"

# ---- DOWNLOAD ----
echo "Downloading data archive..."
wget -O "${ZIP_NAME}" "${ZIP_URL}"

# ---- UNPACK ----
echo "Unpacking data..."
unzip -o "${ZIP_NAME}"

# ---- CLEANUP ----
echo "Removing archive..."
rm "${ZIP_NAME}"

echo "Done. data_fire/ and data_flood/ are ready."
