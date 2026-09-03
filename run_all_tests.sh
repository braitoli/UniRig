#!/bin/bash
set -e

echo "=================================================="
echo "🧪 Running UniRig 3D Statue Pipeline & API Tests"
echo "=================================================="

export PYTHONNOUSERSITE=1
/home/braitoli/miniconda/envs/unirig312/bin/python -m unittest tests/test_statue_pipeline.py

echo ""
echo "=================================================="
echo "🌐 Running Headless Browser E2E Tests (Chromium CDP)"
echo "=================================================="

# Ensure server is online
/home/braitoli/miniconda/envs/unirig312/bin/python /home/braitoli/workspace/namnh/code/poc/UniRig/start_server.py

node tests/test_browser_e2e.mjs

echo ""
echo "=================================================="
echo "🎨 Running Preset Chips (Ảnh Mẫu Nhanh) E2E Tests"
echo "=================================================="

node tests/test_preset_chips_e2e.mjs

echo ""
echo "=================================================="
echo "✅ ALL TESTS PASSED SUCCESSFULLY! (100% OK)"
echo "=================================================="
