#!/bin/bash
set -e

# Parse CPU and memory binding arguments
sample_size=${sample_size:-"100"}
port=${port:-"8080"}
hostname=${hostname:-"10.86.18.6"}
topic=${topic:-"1P_VIT"}

echo "Starting vit_loadgen with Sample Size:${sample_size}, Port:${port}, Hostname:${hostname} and Topic:${topic}"

# Run TorchServe with dynamic numactl settings
exec python3 vit_loadgen.py --sample_size ${sample_size} --port ${port} --hostname ${hostname} --topic ${topic}
