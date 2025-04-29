

export CUDA_VISABLE_DEVICES=0
# export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
# cuda-memcheck python bench.py
python bench.py

