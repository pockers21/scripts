import sys
import time
import torch
import gc
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM
from accelerate import Accelerator, ProfileKwargs
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import os
from memory_profiler import profile



model_base = "/data/sonald/ai_models/model_weights/"
model_name = "Qwen2-0.5B-Instruct"
model_save = model_base + model_name

model =  None

#@profile()

load_list = []
eval_list = []
def execute(batch_size, max_seq_length):
    batch_size, max_seq_length = batch_size, max_seq_length
    print(batch_size, max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(model_save, torch_dtype=torch.bfloat16)

    device = torch.device("cpu")
    model = model.to(device)

    model = model.eval()
    print(model.dtype)

    inputs = torch.randint(1,15000,(batch_size, max_seq_length)).long().to(device)

    #accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = round(memory_info.rss / (1024 ** 3), 2)
    load_list.append(memory_usage_gb)
    print(f"LoadMemoryUsage: {memory_usage_gb} GB")
    model(inputs)
    memory_info = process.memory_info()
    memory_usage_gb = round(memory_info.rss / (1024 ** 3), 2)
    eval_list.append(memory_usage_gb)
    print(f"EvalMemoryUsage: {memory_usage_gb} GB")
    del model
    del inputs
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == '__main__':
    execute(int(sys.argv[1]), int(sys.argv[2]))
    print(f'load_list:{load_list}')
    print(f'eval_list:{eval_list}')
