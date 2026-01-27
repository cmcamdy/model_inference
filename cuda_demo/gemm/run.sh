

# nsys profile --trace=cuda,cudnn,cublas -o GEMM3.nsys-rep python bench.py
# nsys profile  --trace=cuda,cudnn,cublas,nvtx,osrt --force-overwrite true --stats=true -o GEMM3.nsys-rep  python bench.py
# nsys profile --capture cuda --profile-from-start off  -o GEMM3.nsys-rep  python bench.py
# nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi  --force-overwrite true -o GEMM3.nsys-rep python bench.py


# /usr/local/NVIDIA-Nsight-Compute-2023.3/ncu --section={LaunchStats,InstructionStats,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Tables,SourceCounters,SpeedOfLight,Occupancy} \
#         --target-processes all -f -o smem_gemm_blocktiling_2d python bench.py


# /usr/local/NVIDIA-Nsight-Compute-2023.3/ncu --section={LaunchStats,InstructionStats,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Tables,SourceCounters,SpeedOfLight,Occupancy} \
#         --target-processes all -f -o gemm_all_1024 python bench.py

/usr/local/NVIDIA-Nsight-Compute-2023.3/ncu --section={LaunchStats,InstructionStats,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Tables,SourceCounters,SpeedOfLight,Occupancy} \
        --target-processes all -f -o gemm_smem_naive python bench.py