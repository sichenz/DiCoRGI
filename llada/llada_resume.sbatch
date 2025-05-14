#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=llada_resume
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=llada_resume.out

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

# Create checkpoint directory
mkdir -p checkpoints

# Run the Python script with error handling and checkpointing
singularity exec --nv \
    --overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate arc
        
        # Check if we need to resume from a checkpoint
        if [ -f 'llada_base_results/all_results.json' ]; then
            echo 'Found previous results, resuming...'
            python llada_resume.py 2 --batch-size 20 --num-tasks 144 --resume 2>&1
        else
            echo 'Starting fresh run for last 144 tasks...'
            python llada_resume.py 2 --batch-size 20 --num-tasks 144 2>&1
        fi
    "


# Display summary if available
if [ -f "llada_base_results/summary.json" ]; then
    echo "Summary:"
    cat llada_base_results/summary.json
fi

# Check if we need to report partial completion
if [ -f "llada_base_results/intermediate_results.json" ]; then
    echo "Intermediate results saved"
    completed_count=$(grep -o '"status": "success"' llada_base_results/intermediate_results.json | wc -l)
    echo "Completed $completed_count tasks successfully"
fi