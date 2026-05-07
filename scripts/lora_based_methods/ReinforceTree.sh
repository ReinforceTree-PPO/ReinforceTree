#!/bin/bash

# Script to run Tree_LoRA with different lora_depth values

# Run cleanup script
echo "Running cleanup script..."
bash scripts/cleanup_ports.sh

# Activate virtual environment
source /home/mlo/garima/Reinforcetree/treelora_env/bin/activate

# Set up CUDA library paths for bitsandbytes
export LD_LIBRARY_PATH=/home/mlo/garima/Reinforcetree/treelora_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/mlo/garima/Reinforcetree/treelora_env/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/mlo/garima/Reinforcetree/treelora_env/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/mlo/garima/Reinforcetree/treelora_env/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

gpu_nodes="0,1,2,3"
# Remove fake CUDA - let it use real CUDA
# export CUDA_HOME=/tmp/fake_cuda
export DS_BUILD_OPS=1
export DS_BUILD_FUSED_ADAM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name="Llama-3.2-1B-Instruct"
#model_name="Llama-2-7b-chat"
#model_name="Llama-3.1-8B-Instruct"
#model_name="Llama-3.2-1B-Instruct"
#model_name="Qwen2.5-7B-Instruct"
#model_name="Mistral-7B-Instruct-v0.3"
#model_name="gemma-2b-it"

epochs="2,1,3,2,1,2,2,3"
reg=0.5

# Function to find an available port
find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# Array of lora_depth values to test
lora_depths=(8 16 32 64)

# Loop through each lora_depth
for lora_depth in "${lora_depths[@]}"
do
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
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --max_prompt_len 512 \
        --max_ans_len 256 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --num_train_epochs "$epochs" \
        --gradient_accumulation_steps 16 \
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
        echo "Training failed for lora_depth=$lora_depth, skipping inference..."
        continue
    fi
    
    # Verify model checkpoints exist
    if [ ! -f "./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/0/adapter_config.json" ]; then
        echo "Model checkpoint not found for lora_depth=$lora_depth, skipping inference..."
        continue
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
        echo "Inference failed for lora_depth=$lora_depth, skipping result collection..."
        continue
    fi

    # Collect results
    echo "Start collecting results for lora_depth=$lora_depth..."
    source /home/mlo/garima/Reinforcetree/treelora_env/bin/activate
    python inference/collect_results.py \
        --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
        --data_path ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions
    
    # Check if result collection succeeded and copy with depth-specific name
    if [ $? -eq 0 ] && [ -f "./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions/final_result.txt" ]; then
        cp ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/predictions/final_result.txt \
           ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/final_result_depth${lora_depth}.txt
        echo "Results saved to: ./outputs_LLM-CL/depth${lora_depth}_reg${reg}/$model_name/Tree_LoRA_$now/final_result_depth${lora_depth}.txt"
    else
        echo "Result collection failed for lora_depth=$lora_depth"
    fi
    
    echo "Completed experiment with lora_depth=$lora_depth"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
