import sys
import time
import torch
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM
from accelerate import Accelerator, ProfileKwargs
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoConfig

def restrict_gpu_upper_bound():
    memory_fraction = 0.3  # 例如，设置为 50%

    torch.cuda.set_per_process_memory_fraction(memory_fraction)

    total_memory_bytes = torch.cuda.get_device_properties(1).total_memory

    available_memory_bytes = total_memory_bytes * memory_fraction

    available_memory_gb = available_memory_bytes / (1024 ** 3)

    print(f"Available memory for the process: {available_memory_gb:.2f} GB")


    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=1)  # 例如，限制为 50%
dataset_paths = {
    1:  ("glm-4-9b-chat",9),
    2: ("Qwen2-1.5B-Instruct",1.5),
    3: ("Qwen2-0.5B-Instruct",0.5),
    4: ("Qwen2-7B-Instruct",7),
    5: ("Meta-Llama-3-8B-Instruct",8),
    6: ("deepseek-coder-1.3b-instruct",1.3),
    7: ("deepseek-coder-7b-instruct-v1.5",1.5),
    8: ("Phi-3-small-8k-instruct",0),
    9: ("bge-large-zh-v1.5",1.5),
    10: ("Qwen2-0.5B-Instruct-GPTQ-Int8",0),
    11: ("Qwen2-1.5B-Instruct-GPTQ-Int8",0),
    12: ("Qwen2-7B-Instruct-GPTQ-Int8",0)
}

print("请选择一个数据集：")
for key, value in dataset_paths.items():
    print(f"{key}. {value[0]}")

choice = int(input("请输入你的选择（数字）："))

if choice in dataset_paths:
    print(f"你选择的数据集路径是：{dataset_paths[choice][0]}")
else:
    print("输入的数字不在可选范围内，请重新输入！")


#restrict_gpu_upper_bound()

batch_size, max_seq_length = int(sys.argv[1]), int(sys.argv[2])
print(batch_size, max_seq_length)
model_base = "/hy-tmp/"
model_name = dataset_paths[choice][0]
model_save = model_base + model_name

model =  None

#model = AutoModelForCausalLM.from_pretrained(model_save, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_save, trust_remote_code=True)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
model = model.to(device)

model = model.eval()


for name, param in model.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape} type:{param.dtype}")


for name, param in model.named_parameters():
        print(f'param dtype: {param.dtype}')

        if 'weight' in name:
            print(f"Parameter name: {name}, shape: {param.shape}, min value: {param.min()}, max value: {param.max()}")
    


inputs = torch.randint(1,15000,(batch_size, max_seq_length)).long().to(device)

input_list = [inputs]

torch.cuda.reset_peak_memory_stats()
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True
    ) as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="flops", row_limit=10))
print(f'batch_size:{batch_size:} max_seq_length:{max_seq_length:}')
print(f'model_name:{model_name}')
