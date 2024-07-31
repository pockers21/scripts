#!/bin/bash

# 检查传入的参数是否正确
if [ "$#" -ne 3 ] || [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

load_list=()
eval_list=()

declare -A dataset_paths
dataset_paths[1]="glm-4-9b-chat"
dataset_paths[2]="Qwen2-1.5B-Instruct"
dataset_paths[3]="Qwen2-0.5B-Instruct"
dataset_paths[4]="Qwen2-7B-Instruct"
dataset_paths[5]="Meta-Llama-3-8B-Instruct"
dataset_paths[6]="deepseek-ai/deepseek-coder-1.3b-instruct"
dataset_paths[7]="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
dataset_paths[8]="Phi-3-small-8k-instruct"
dataset_paths[9]="bge-large-zh-v1.5"

echo "Available datasets:"
for i in {1..9}; do
    echo "$i: ${dataset_paths[$i]}"
done
read -p "Enter the dataset number (1-9): " user_input

if ! [[ "$user_input" =~ ^[1-9]$ ]]; then
    echo "Invalid input. Please enter a number between 1 and 9."
    exit 1
fi

data_set_path="${dataset_paths[$user_input]}"


for i in {1..3}; do
    echo "Execution $i:"
    if [ "$1" == "cpu" ]; then
        output=$(python hf-flops_cpu.py $2 $3 $data_set_path)
    else
        output=$(python hf-flops_gpu.py $2 $3 $data_set_path)
    fi
    echo "$output"
    load_memory_usage=$(echo "$output" | grep -o -E "LoadMemoryUsage: [0-9.]+ GB" | awk '{print $2}')
    eval_memory_usage=$(echo "$output" | grep -o -E "EvalMemoryUsage: [0-9.]+ GB" | awk '{print $2}')

    load_list+=("$load_memory_usage")
    eval_list+=("$eval_memory_usage")

    echo "LoadMemoryUsage: $load_memory_usage"
    echo "EvalMemoryUsage: $eval_memory_usage"
    echo
done

echo "Load Memory Usage List: ${load_list[@]}"
echo "Eval Memory Usage List: ${eval_list[@]}"

load_avg=$( printf "%0.2f" "$(
    for i in "${load_list[@]}"; do
        echo "$i" 
    done | awk '{ sum += $1 } END { print sum/NR }'
)")
eval_avg=$( printf "%0.2f" "$(
    for i in "${eval_list[@]}"; do
        echo "$i"
    done | awk '{ sum += $1 } END { print sum/NR }'
)")

echo "Average Load Memory Usage: $load_avg GB"
echo "Average Eval Memory Usage: $eval_avg GB"
