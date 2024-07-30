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


restrict_gpu_upper_bound()

batch_size, max_seq_length = int(sys.argv[1]), int(sys.argv[2])
print(batch_size, max_seq_length)
model_base = "/data/sonald/ai_models/model_weights/"
model_name = "Qwen2-0.5B-Instruct"
model_save = model_base + model_name

model =  None

#model = AutoModelForCausalLM.from_pretrained(model_save,torch_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained(model_save, torch_dtype=torch.bfloat16)
#model = torch.compile(model)

device = torch.device("cuda:1")
torch.cuda.set_device(device)
torch.cuda.reset_max_memory_allocated(device)
origin_bytes = torch.cuda.memory_allocated(device=device)
print(f'origin_bytes:{int(origin_bytes)/1024/1024/1024}')
model = model.to(device)

load_bytes = torch.cuda.memory_allocated(device=device)
load_bytes = (int(load_bytes))/(1024 ** 3)



print(f'LoadMemoryUsage: {load_bytes} GB')

model = model.eval()
print(model.dtype)

inputs = torch.randint(1,15000,(batch_size, max_seq_length)).long().to(device)

input_list = [inputs]

profile_kwargs = ProfileKwargs(
        #with_flops=True,
        profile_memory=True,
        activities=["cuda","cpu"],
        #record_shapes=True
)

#accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

torch.cuda.reset_peak_memory_stats()
memory_before = torch.cuda.memory_allocated()
model(inputs)
memory_after = torch.cuda.memory_allocated()
memory_usage = (memory_after - memory_before) / (1024 ** 3)  # in MB
print(f'before:{memory_before/(1024**3)}')
print(f'after:{memory_after/(1024**3)}')
#print(f'memory_usage:{memory_usage}')
print(f'EvalMemoryUsage: {int(torch.cuda.max_memory_allocated())/(1024**3)} GB')
#print(prof.key_averages().table(sort_by="flops", row_limit=10))
print(int(torch.cuda.max_memory_allocated())/(1024**3))
