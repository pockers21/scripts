import sys
import time
import torch
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM
from accelerate import Accelerator, ProfileKwargs
from torch.profiler import profile, record_function, ProfilerActivity

def restrict_gpu_upper_bound():
    memory_fraction = 0.3  # 例如，设置为 50%

    torch.cuda.set_per_process_memory_fraction(memory_fraction)

    total_memory_bytes = torch.cuda.get_device_properties(1).total_memory

    available_memory_bytes = total_memory_bytes * memory_fraction

    available_memory_gb = available_memory_bytes / (1024 ** 3)

    print(f"Available memory for the process: {available_memory_gb:.2f} GB")


    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=1)  # 例如，限制为 50%


#restrict_gpu_upper_bound()

batch_size, max_seq_length = int(sys.argv[1]), int(sys.argv[2])
print(batch_size, max_seq_length)
model_base = "/hy-tmp/"
model_name = sys.argv[3]
model_save = model_base + model_name

model =  None
device = torch.device("cuda:0")
origin_gbytes = torch.cuda.memory_allocated(device=device) /(1024**3)
print(f'origin_gbytes:{origin_gbytes} GBytes')

#model = AutoModelForCausalLM.from_pretrained(model_save,torch_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(model_save, torch_dtype=torch.bfloat16, trust_remote_code=True)
#model = torch.compile(model)


torch.cuda.set_device(device)
torch.cuda.reset_max_memory_allocated(device)

model = model.to(device)

loaded_gbytes = int(torch.cuda.memory_allocated(device=device))/(1024**3)
loaded_gbytes -= origin_gbytes

print(f'LoadMemoryUsage: {loaded_gbytes} GB')

model = model.eval()
print(model.dtype)

inputs = torch.randint(1,15000,(batch_size, max_seq_length)).long().to(device)

input_list = [inputs]

#accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

torch.cuda.reset_peak_memory_stats()
#memory_before = torch.cuda.memory_allocated()
model(inputs)
#memory_after = torch.cuda.memory_allocated()
#memory_usage = (memory_after - memory_before) / (1024 ** 3)  # in MB
#print(f'before:{memory_before/(1024**3)}')
#print(f'after:{memory_after/(1024**3)}')
max_memory_allocated = int(torch.cuda.max_memory_allocated())/(1024**3) - origin_gbytes
print(f'EvalMemoryUsage: {max_memory_allocated} GB')
#print(prof.key_averages().table(sort_by="flops", row_limit=10))
print(int(torch.cuda.max_memory_allocated())/(1024**3))

print(f'batch_size:{batch_size:} max_seq_length:{max_seq_length:}')
print(f'model_name:{model_name}')
