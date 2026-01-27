# nvcc  -O0 -o cuda_kernel_performance cuda_kernel_performance.cu
# # nsys profile --force-overwrite=true  --trace=cuda,nvtx  -o cuda_kernel_performance.nsys-rep ./cuda_kernel_performance
# nsys profile --trace=cuda,cudnn,cublas,nvtx,osrt --force-overwrite true --stats=true -o cuda_kernel_performance.nsys-rep ./cuda_kernel_performance
# # ./cuda_kernel_performance


# Linux需调整perf权限
# echo 2 > /proc/sys/kernel/perf_event_paranoid  # 临时设置
# echo 'kernel.perf_event_paranoid=2' >> /etc/sysctl.conf  # 永久生效:cite[9]

nvcc -o test test.cu
nsys profile --stat true --trace=cuda,cudnn,cublas,nvtx --force-overwrite true -o test.nsys-rep ./test
 

# nvcc test.cu -o test -lcudart -lnvToolsExt
# nsys profile --trace=nvtx,cuda --force-overwrite true -o test.nsys-rep ./test

# export PATH=$PATH:/usr/local/cuda-11.7/nsight-compute-2022.2.1/bin

nsys profile \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite true\
  -o gemm_profile \
  python test_torch_profiling.py