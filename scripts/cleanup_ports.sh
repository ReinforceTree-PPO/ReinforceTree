#!/bin/bash

# Script to cleanup ports and processes

echo "Cleaning up TreeLoRA processes and ports..."

# Kill deepspeed processes
pkill -9 -f "deepspeed" 2>/dev/null || true
pkill -9 -f "training/main.py" 2>/dev/null || true
pkill -9 -f "inference/infer" 2>/dev/null || true

# Find and kill processes using common ports
for port in 25011 25012 25013 25014 25015; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid using port $port"
        kill -9 $pid 2>/dev/null || true
    fi
done

echo "Cleanup complete. Waiting 3 seconds..."
sleep 3
echo "Ready to run experiments."
