nvcc  -O0 -o cuda_kernel_performance cuda_kernel_performance.cu
# nsys profile --force-overwrite=true  --trace=cuda,nvtx  -o cuda_kernel_performance.nsys-rep ./cuda_kernel_performance
nsys profile --trace=cuda,cudnn,cublas,nvtx --system-trace true --force-overwrite true  -o cuda_kernel_performance.nsys-rep ./cuda_kernel_performance
# ./cuda_kernel_performance

nvcc -o test test.cu
nsys profile --trace=cuda,cudnn,cublas,nvtx --force-overwrite true -o test.nsys-rep ./test
 