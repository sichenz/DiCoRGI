#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=128GB
#SBATCH --partition=a100_2
#SBATCH --job-name=llada_sft_parallel
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llada_sft_parallel_%j.out

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $(nproc)"

# Set up distributed training environment
export WORLD_SIZE=2
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Set CPU affinity for better performance
export OMP_NUM_THREADS=4

# Run distributed training
singularity exec --nv \
--overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
/bin/bash -c "
source /ext3/env.sh; 
conda activate arc; 
cd /scratch/sz4972/DiCoRGI/llada;
python -m torch.distributed.launch --nproc_per_node=2 llada_sft_parallel.py \
    --epochs 5 \
    --batch-size 4 \
    --lr 1e-5 \
    --n-gpus 2 \
    --n-cpus 8
"

echo "Job completed at $(date)"