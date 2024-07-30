#!/bin/bash

if [ "$#" -ne 3 ] || [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

load_list=()
eval_list=()

for i in {1..3}; do
    echo "Execution $i:"
    if [ "$1" == "cpu" ]; then
        output=$(python hf-flops_cpu.py $2 $3)
    else
        output=$(python hf-flops_gpu.py $2 $3)
    fi
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
