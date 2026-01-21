#!/bin/bash
# Phase 2: Full Grid Test with Optimized Configuration
#
# Based on Phase 1 findings:
# - min-mag-offset=0.0 gives +48% completeness improvement
# - No increase in false positives
# - Other changes (GP modifications, logbf threshold, mag grid) showed no benefit
#
# This script runs a comprehensive comparison:
# 1. Baseline (current production config)
# 2. Optimized (min-mag-offset=0.0)

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 2: Full Grid Test with Optimized Configuration"
echo "==================================================================="
echo ""
echo "Parameter space:"
echo "  Amplitude: 0.05 - 5.0 mag (fractional depth 0.045 - 0.99)"
echo "  Duration:  1 - 300 days (log-uniform sampling)"
echo "  Mag grid:  25 points (0.2 mag resolution)"
echo "  Trials:    100 x 100 x 100 = 1M injections per config"
echo ""

# Test 2-1: Baseline (current production configuration)
echo "[1/2] Running BASELINE (current production config)..."
echo "  - Injection test..."
python -m malca.injection --run-tag "2_baseline" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 100 \
  --dur-min 1 --dur-max 300 --dur-steps 100 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "2_baseline" \
  --sample-size 10000 \
  --mag-points 25 \
  --workers 50

# Test 2-2: Optimized (min-mag-offset=0.0)
echo ""
echo "[2/2] Running OPTIMIZED (min-mag-offset=0.0)..."
echo "  - Injection test..."
python -m malca.injection --run-tag "2_optimized" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 100 \
  --dur-min 1 --dur-max 300 --dur-steps 100 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.0

echo "  - Detection rate (false positive measurement)..."
python -m malca.detection_rate --run-tag "2_optimized" \
  --sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.0

echo ""
echo "==================================================================="
echo "Phase 2 Complete!"
echo "==================================================================="
echo ""
echo "Results:"
echo "  Injection (completeness):"
echo "    - output/injection/*_2_baseline/"
echo "    - output/injection/*_2_optimized/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output/detection_rate/*_2_baseline/"
echo "    - output/detection_rate/*_2_optimized/"
echo ""
echo "Next: Run analyze_phase2.py to compare results"
echo "==================================================================="
