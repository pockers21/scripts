export PYTHONPATH=/root/EAGLE/eagle
 
python  gen_baseline_answer_qwen2.py --ea-model-path /hy-tmp/yuhuili/EAGLE-Qwen2-7B-Instruct --base-model-path /hy-tmp/Qwen2-7B-Instruct
python  gen_ea_answer_qwen2.py  --ea-model-path /hy-tmp/yuhuili/EAGLE-Qwen2-7B-Instruct --base-model-path /hy-tmp/Qwen2-7B-Instruct