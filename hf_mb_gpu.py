import sys
import time
import torch
import gc
from transformers import AutoTokenizer
import time
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator, ProfileKwargs
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import os
from colorama import Fore, Style
from transformers import GPTQConfig





model =  None
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
    12: ("Qwen2-7B-Instruct-GPTQ-Int8",0),
    13: ("Qwen2-0.5B-Instruct-GPTQ-Int4",0)
}

print("请选择一个数据集：")
for key, value in dataset_paths.items():
    print(f"{key}. {value[0]}")

choice = int(input("请输入你的选择（数字）："))

if choice in dataset_paths:
    print(f"你选择的数据集路径是：{dataset_paths[choice][0]}")
else:
    print("输入的数字不在可选范围内，请重新输入！")

model_base = "/hy-tmp/"
model_name = dataset_paths[choice][0]
model_save = model_base + model_name


load_list = []
eval_list = []

def caculate_kv(output):
    past_key_values = output.past_key_values
    total_bytes = 0 
    layercnt=0
    tensorcnt=0
    for layer in past_key_values:
        layercnt += 1
        for tensor in layer:
            tensorcnt += 1
            total_bytes += tensor.element_size() * tensor.nelement()
    print("KV Cache Total Bytes:", total_bytes)
    return total_bytes


def execute1(batch_size, max_seq_length, load_bits):
    print(batch_size, max_seq_length)
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    param = {
        "pretrained_model_name_or_path":model_save,
        "trust_remote_code":True,
        "use_cache":True,
        "torch_dtype" : "auto"
    } 
    if load_bits == 4:
        #param["torch_dtype"] = torch.int4
        param['quantization_config'] = GPTQConfig(
            bits=4,
            disable_exllama=True)
    elif load_bits == 8:
        #param["torch_dtype"] = torch.int8
        param['quantization_config'] = GPTQConfig(
            bits=8,
            )
    elif load_bits == 16:
        #param["torch_dtype"] = torch.bfloat16
        pass


    model = AutoModelForCausalLM.from_pretrained(**param)
    print(f'model type:{model.dtype}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


    tokenizer = AutoTokenizer.from_pretrained(model_save)
   
    model = model.to(device)
    print(f'model.device:{model.device}')
    print(f'model.get_memory_footprint:{model.get_memory_footprint()}')
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape} type:{param.dtype}")
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Parameter name: {name}, shape: {param.shape}, min value: {param.min()}, max value: {param.max()}")
    """
    
    model.eval()
    
    text = "你好，今天感觉如何？"  
    text = [text] * batch_size
    
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"]
    start_time_first_token = time.time()
    
    with torch.no_grad():
        #inputs = torch.randint(1,15000,(batch_size, 6)).long().to(device)
        #model(inputs)

        output_first_token = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1  # 生成一个token
            #max_length=1
        )

    #print(f'output_first_token shape:{output_first_token.shape}')
    end_time_first_token = time.time()
    
    ttft = end_time_first_token - start_time_first_token
    print(Fore.RED + f"TTFT: {ttft:.4f} seconds"  + Style.RESET_ALL)


    print(output_first_token)
    generated_token = output_first_token[:, -1].unsqueeze(1)
    print(f'input_ids: shape:{input_ids.shape}')
    print(f'generated_token: shape:{generated_token.shape}')
    print(f'mask shape:{attention_mask.shape}')
    input_ids = torch.cat([input_ids, generated_token], dim=1)
    
    start_time_remaining_tokens = time.time()
    attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype)), dim=-1) 
    with torch.no_grad():
        print(f'input_ids:{input_ids.shape}')
        print(f'attention_mask:{attention_mask.shape}')
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=max_seq_length,
            max_length=max_seq_length,  
            return_dict_in_generate=True
        )
    
    end_time_remaining_tokens = time.time()
    
    total_time_remaining_tokens = end_time_remaining_tokens - start_time_remaining_tokens
    
    total_tokens = output.sequences.numel()
    tpot = total_time_remaining_tokens/total_tokens if total_time_remaining_tokens > 0 else float('inf')
    print(f'geneate {total_tokens} tokens')
    print(Fore.RED + f"TPOT: {tpot:.4f} " + Style.RESET_ALL)
    
    print(f'Shape of output sequences: {output.sequences.shape}')
    total_bytes = caculate_kv(output)
    total_gbytes = total_bytes/(1024**3)
    print(Fore.RED + f'kv_cache size:{total_gbytes}' + Style.RESET_ALL)
    bytes_per_param = load_bits/4

    total_gbytes += (total_params*bytes_per_param) / (1024**3)
    print(f'total_gbytes:{total_gbytes}')
    print(Fore.RED +  f'memory bandwidh: {total_gbytes/tpot}' + Style.RESET_ALL)


    
    del model
    gc.collect()
    torch.cuda.empty_cache()

def validate_param():
    args = sys.argv[1:]
    print(f'len : {len(args)}')
    len_args = len(args)
    if len_args != 3:
        raise Exception('args count should be 3, batch size, max seq len, model load bits')
    bs_seq_len_set = {(1,128), (4,128), (1,2048)}
    param_tuple = (int(sys.argv[1]), int(sys.argv[2]))
    if param_tuple not in bs_seq_len_set:
        raise Exception(f'{param_tuple} not in set {bs_seq_len_set}')
    if int(sys.argv[3]) not in {4,8,16}:
        raise Exception(f'{int(sys.argv[3])} not in {{4, 8,16}}')

if __name__ == '__main__':
    validate_param()
    execute1(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
