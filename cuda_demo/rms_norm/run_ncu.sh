export TORCH_CUDA_ARCH_LIST=Ampere
/usr/local/NVIDIA-Nsight-Compute-2023.3/ncu --section={LaunchStats,InstructionStats,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Tables,SourceCounters,SpeedOfLight,Occupancy} \
        --target-processes all -f -o rms_norm_naive python rms_norm.py