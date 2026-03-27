#!/usr/bin/env bash

set -euo pipefail

SCANNET_REPO_URL="https://github.com/ScanNet/ScanNet.git"

# Check pixi environment activation.
if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
    echo "Error: PIXI_PROJECT_ROOT is not set. Please activate the pixi environment."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required but not found in PATH."
    exit 1
fi

if ! command -v make >/dev/null 2>&1; then
    echo "Error: make is required but not found in PATH."
    exit 1
fi

SCANNET_DIR="${PIXI_PROJECT_ROOT}/ScanNet"
SEGMENTATOR_DIR="${SCANNET_DIR}/Segmentator"

if [ -d "${SCANNET_DIR}" ]; then
    echo "ScanNet directory already exists. Skipping clone."
else
    git clone "${SCANNET_REPO_URL}" "${SCANNET_DIR}"
fi

if [ ! -d "${SEGMENTATOR_DIR}" ]; then
    echo "Error: Segmentator directory not found at ${SEGMENTATOR_DIR}."
    exit 1
fi

pushd "${SCANNET_DIR}" >/dev/null
git fetch --tags --force
# git checkout "${SCANNET_COMMIT}"
popd >/dev/null

pushd "${SEGMENTATOR_DIR}" >/dev/null
make
popd >/dev/null

echo "ScanNet Segmentator setup complete."