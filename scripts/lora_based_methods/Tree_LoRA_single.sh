#!/bin/bash

# Minimal script - runs only ONE lora_depth to save disk space

# Run cleanup script
echo "Running cleanup script..."
bash scripts/cleanup_ports.sh

# Activate virtual environment
source /data1/anchal/garima/TreeLoRA/treelora_env/bin/activate

# Set up CUDA library paths for bitsandbytes
export LD_LIBRARY_PATH=/data1/anchal/garima/TreeLoRA/treelora_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data1/anchal/garima/TreeLoRA/treelora_env/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data1/anchal/garima/TreeLoRA/treelora_env/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data1/anchal/garima/TreeLoRA/treelora_env/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

gpu_nodes="0"
export DS_BUILD_OPS=1
export DS_BUILD_FUSED_ADAM=1

model_name="Llama-2-7b-chat"
epochs="2,1,3,2,1,2,2,3"
reg=0.5

# Function to find an available port
find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# ONLY ONE lora_depth to save space
lora_depth=16

now=$(date +"%m%d_%H%M%S")

# Find available ports for this experiment
master_port=$(find_free_port)
echo "Using master port: $master_port"

echo "=========================================="
echo "Starting experiment with lora_depth=$lora_depth"
echo "=========================================="

# Kill any lingering processes on the port
pkill -f "master_port.*$master_port" 2>/dev/null || true
sleep 2

# Train
echo "Start training with lora_depth=$lora_depth..."
deepspeed --include=localhost:$gpu_nodes --master_port $master_port training/main.py  \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 512 \
    --max_ans_len 256 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --num_train_epochs "$epochs" \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method Tree_LoRA \
    --lora_depth $lora_depth \
    --output_dir ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now \
    --reg $reg \
    --use_ppo \
    --ppo_lr 3e-4 \
    --ppo_clip 0.2

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "Training failed for lora_depth=$lora_depth"
    exit 1
fi

# Verify model checkpoints exist
if [ ! -f "./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/0/adapter_config.json" ]; then
    echo "Model checkpoint not found for lora_depth=$lora_depth"
    exit 1
fi

# Create predictions directory
mkdir -p ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions

# Find new port for inference
inference_port=$(find_free_port)
echo "Using inference port: $inference_port"

# Inference
echo "Start inference with lora_depth=$lora_depth..."
python inference/infer_multi_command.py  \
    --gpus $gpu_nodes \
    --start_round 0 \
    --master_port $inference_port \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method Tree_LoRA \
    --inference_output_path ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions

# Check if inference succeeded
if [ $? -ne 0 ]; then
    echo "Inference failed for lora_depth=$lora_depth"
    exit 1
fi

# Collect results
echo "Start collecting results for lora_depth=$lora_depth..."
python inference/collect_results.py \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --data_path ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions

echo "=========================================="
echo "Experiment completed successfully!"
echo "=========================================="
