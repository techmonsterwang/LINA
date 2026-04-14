#!/bin/bash

# Script to run FLOPs test
# This script runs the FLOPs test for Linear Attention and Full Attention models

echo "Starting FLOPs test..."
echo "======================================"

# Change to the script directory
cd "$(dirname "$0")"

# Run the test
python test_flops.py

echo ""
echo "======================================"
echo "FLOPs test completed!"
echo "Check the following files for results:"
echo "  - flops_results.txt (text results)"
echo "  - flops_comparison.pdf (bar chart)"
echo "  - flops_comparison.svg (bar chart)"
echo "  - flops_comparison.png (bar chart)"

