#!/bin/bash
# Phase 1C: Diagnose Deep-Short Combination Bottleneck
# Tests: baseline, extended_mag_grid, deep_mag_grid
# Also runs detection_rate to measure false positive rates

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 1C: Deep-Short Combination Bottleneck Diagnostic"
echo "==================================================================="

# Test 1C-1: Baseline (current configuration)
echo ""
echo "[1/3] Running baseline (current config)..."
echo "  - Injection test (100×100 grid, 100 inj/cell = 1M trials)..."
python -m malca.injection --run-tag "1c_baseline" \
  --amp-min 1.0 --amp-max 3.0 --amp-steps 100 \
  --dur-min 1 --dur-max 50 --dur-steps 100 \
  --n-injections-per-grid 100 \
  --workers 40

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1c_baseline" \
  --sample-size 10000 \
  --workers 40

# Test 1C-2: Extended mag grid (to mag 18)
echo ""
echo "[2/3] Running with extended mag grid (--mag-max-dip 18.0)..."
echo "  - Injection test (100×100 grid, 100 inj/cell = 1M trials)..."
python -m malca.injection --run-tag "1c_extended_mag_grid" \
  --amp-min 1.0 --amp-max 3.0 --amp-steps 100 \
  --dur-min 1 --dur-max 50 --dur-steps 100 \
  --n-injections-per-grid 100 \
  --workers 40 \
  --mag-max-dip 18.0

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1c_extended_mag_grid" \
  --sample-size 10000 \
  --workers 40 \
  --mag-max-dip 18.0

# Test 1C-3: Very deep mag grid (to mag 20)
echo ""
echo "[3/3] Running with very deep mag grid (--mag-max-dip 20.0)..."
echo "  - Injection test (100×100 grid, 100 inj/cell = 1M trials)..."
python -m malca.injection --run-tag "1c_deep_mag_grid" \
  --amp-min 1.0 --amp-max 3.0 --amp-steps 100 \
  --dur-min 1 --dur-max 50 --dur-steps 100 \
  --n-injections-per-grid 100 \
  --workers 40 \
  --mag-max-dip 20.0

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1c_deep_mag_grid" \
  --sample-size 10000 \
  --workers 40 \
  --mag-max-dip 20.0

echo ""
echo "==================================================================="
echo "Phase 1C Complete!"
echo "Results in:"
echo "  Injection (completeness):"
echo "    - output/injection/1c_baseline/"
echo "    - output/injection/1c_extended_mag_grid/"
echo "    - output/injection/1c_deep_mag_grid/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output/detection_rate/1c_baseline/"
echo "    - output/detection_rate/1c_extended_mag_grid/"
echo "    - output/detection_rate/1c_deep_mag_grid/"
echo ""
echo "Next: Compare completeness vs contamination trade-offs"
echo "==================================================================="
