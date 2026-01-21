#!/bin/bash
# Phase 1B: Diagnose Short Duration Bottleneck
# Tests: baseline, short_gp, trend_baseline, masked_gp
# Also runs detection_rate to measure false positive rates

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 1B: Short Duration Bottleneck Diagnostic"
echo "==================================================================="

# Test 1B-1: Baseline (current configuration)
echo ""
echo "[1/4] Running baseline (current config)..."
echo "  - Injection test..."
python -m malca.injection --run-tag "1b_baseline" \
  --amp-min 0.2 --amp-max 0.8 --amp-steps 30 \
  --dur-min 1 --dur-max 100 --dur-steps 50 \
  --n-injections-per-grid 30 \
  --max-trials 45000 \
  --workers 40

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1b_baseline" \
  --sample-size 10000 \
  --workers 40

# Test 1B-2: Shorter GP timescale (~100 days instead of ~2000)
echo ""
echo "[2/4] Running with shorter GP timescale (~100 days)..."
echo "  - Injection test..."
python -m malca.injection --run-tag "1b_short_gp" \
  --amp-min 0.2 --amp-max 0.8 --amp-steps 30 \
  --dur-min 1 --dur-max 100 --dur-steps 50 \
  --n-injections-per-grid 30 \
  --max-trials 45000 \
  --workers 40 \
  --baseline-s0 0.002 --baseline-w0 0.0314

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1b_short_gp" \
  --sample-size 10000 \
  --workers 40 \
  --baseline-s0 0.002 --baseline-w0 0.0314

# Test 1B-3: Use trend baseline (no GP absorption)
echo ""
echo "[3/4] Running with trend baseline (no GP)..."
echo "  - Injection test..."
python -m malca.injection --run-tag "1b_trend_baseline" \
  --amp-min 0.2 --amp-max 0.8 --amp-steps 30 \
  --dur-min 1 --dur-max 100 --dur-steps 50 \
  --n-injections-per-grid 30 \
  --max-trials 45000 \
  --workers 40 \
  --baseline-func trend

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1b_trend_baseline" \
  --sample-size 10000 \
  --workers 40 \
  --baseline-func trend

# Test 1B-4: Use masked GP
echo ""
echo "[4/4] Running with masked GP..."
echo "  - Injection test..."
python -m malca.injection --run-tag "1b_masked_gp" \
  --amp-min 0.2 --amp-max 0.8 --amp-steps 30 \
  --dur-min 1 --dur-max 100 --dur-steps 50 \
  --n-injections-per-grid 30 \
  --max-trials 45000 \
  --workers 40 \
  --baseline-func gp_masked

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "1b_masked_gp" \
  --sample-size 10000 \
  --workers 40 \
  --baseline-func gp_masked

echo ""
echo "==================================================================="
echo "Phase 1B Complete!"
echo "Results in:"
echo "  Injection (completeness):"
echo "    - output/injection/1b_baseline/"
echo "    - output/injection/1b_short_gp/"
echo "    - output/injection/1b_trend_baseline/"
echo "    - output/injection/1b_masked_gp/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output/detection_rate/1b_baseline/"
echo "    - output/detection_rate/1b_short_gp/"
echo "    - output/detection_rate/1b_trend_baseline/"
echo "    - output/detection_rate/1b_masked_gp/"
echo ""
echo "Next: Compare completeness vs contamination trade-offs"
echo "==================================================================="
