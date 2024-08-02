# scripts
host memory usages in load stage were much smaller than model size, so we measure actual size in forward stage.

calculate host/device memory usage:
```
bash run.sh cpu  1 128
```

calculate FLOPs:
```
python hf_flops_cpu.py 1 128 model_name
```
