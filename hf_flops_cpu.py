import sys
import torch

from transformers import AutoModelForCausalLM
from accelerate import Accelerator, ProfileKwargs
import os
from torch.profiler import profile, record_function, ProfilerActivity



model_base = "/hy-tmp/"

model_name = sys.argv[3]
model_save = model_base + model_name

model =  None


load_list = []
eval_list = []
def execute(batch_size, max_seq_length):
    batch_size, max_seq_length = batch_size, max_seq_length
    print(batch_size, max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(model_save, torch_dtype=torch.bfloat16, trust_remote_code=True)

    device = torch.device("cpu")
    model = model.to(device)

    model = model.eval()
    print(model.dtype)

    inputs = torch.randint(1,15000,(batch_size, max_seq_length)).long().to(device)

    #accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True
    ) as prof:
        model(inputs)
    
    print(f'batch_size:{batch_size:} max_seq_length:{max_seq_length:}')
    print(f'model_name:{model_name}')
    print(prof.key_averages().table(sort_by="flops", row_limit=10))

    del model
    del inputs
    torch.cuda.empty_cache()



if __name__ == '__main__':
    execute(int(sys.argv[1]), int(sys.argv[2]))
    print(f'load_list:{load_list}')
    print(f'eval_list:{eval_list}')
