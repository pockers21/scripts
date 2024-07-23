import time
import torch
from calflops import calculate_flops
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM

batch_size, max_seq_length = 1, 2048
model_base = "/hy-tmp/"
model_name = "Qwen2-1.5B-Instruct"
model_save = model_base + model_name
loopcnt = 3
cost_time_list = []
GFLOPs_list = []
GFLOPS_list = []

model =  None

model = AutoModelForCausalLM.from_pretrained(model_save,torch_dtype=torch.bfloat16)
model = model.to(device="cuda").eval()
print(model.dtype)
tokenizer = AutoTokenizer.from_pretrained(model_save)

for i in range(loopcnt):
    #start_time = time.time()
    flops, macs, params, cost_time = calculate_flops(model=model,
                                          input_shape=(batch_size, max_seq_length),
                                          transformer_tokenizer=tokenizer)
    #end_time = time.time()
    #out_elapsed_time = end_time - start_time
    print(f'model path:{model_save}')
    print(f'batch_size：{batch_size}, max_seq_length:{max_seq_length}')
    print(" FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    FLOPs = flops.split(' ')[0]
    suffix = flops.split(' ')[1]
    FLOPs = float(FLOPs)
    if suffix.startswith("TFLOPS"):
        FLOPs *= 1000
    elif suffix.startswith("GFLOPS"):
        pass
    else:
        raise Exception(f'invalid suffix:{suffix}')

    #Llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B 
    print(f'cost time:{cost_time}')
    GFLOPS = FLOPs/cost_time
    #print(f'GFLOPS: {GFLOPs/cost_time}')
    cost_time_list.append(cost_time)
    GFLOPs_list.append(FLOPs)
    GFLOPS_list.append(GFLOPS)
    torch.cuda.empty_cache()
print(f'cost_time_list:{cost_time_list}')
print(f'GFLOPs_list:{GFLOPs_list}')
print(f'GFLOPS_list:{GFLOPS_list}')
