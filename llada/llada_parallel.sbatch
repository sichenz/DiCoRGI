#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=llada_parallel
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=llada_parallel.out

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of CPUs: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"

# Function to handle cleanup on interrupt
cleanup() {
    echo "Caught interrupt signal. Cleaning up..."
    pkill -P $$
    echo "Cleanup complete"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Set PyTorch environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

cd /scratch/sz4972/DiCoRGI/llada

# Run the Python script with error handling
singularity exec --nv \
    --overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate arc
        
        # Run with better error handling and progress logging
        python llada_parallel.py 2 --batch-size 20 2>&1 
        
        # Check for errors and save final status
        if [ \${PIPESTATUS[0]} -eq 0 ]; then
            echo 'SUCCESS' > job_status.txt
        else
            echo 'FAILED' > job_status.txt
        fi
    "

echo "Job finished at $(date)"

# Display summary if available
if [ -f "llada_base_results/summary.json" ]; then
    echo "Summary:"
    cat llada_base_results/summary.json
fi