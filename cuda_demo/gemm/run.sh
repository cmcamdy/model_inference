

export CUDA_VISABLE_DEVICES=0
# export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
# cuda-memcheck python bench.py
python bench.py

# nsys profile --trace=cuda,cudnn,cublas -o GEMM3.nsys-rep python bench.py