#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00      # Increased time limit
#SBATCH --mem=128GB         # Keep this setting as your node supports it
#SBATCH --job-name=llada_sft_opt
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=llada_sft_opt_%j.out

echo "Running on host: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Current working directory: $(pwd)"

# Module purge to clean the environment
module purge

# Create directory for checkpoints
mkdir -p /scratch/sz4972/DiCoRGI/llada/checkpoints

# Run the training script with Singularity
singularity exec --nv \
--overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate arc;
cd /scratch/sz4972/DiCoRGI/llada;

# Print environment info
echo 'Python version:';
python --version;
echo 'CUDA available:';
python -c 'import torch; print(torch.cuda.is_available())';
echo 'CUDA version:';
python -c 'import torch; print(torch.version.cuda)';
echo 'GPU device info:';
python -c 'import torch; print(torch.cuda.get_device_name(0))';
echo 'GPU memory:';
nvidia-smi --query-gpu=memory.total,memory.free --format=csv;

# Run the optimized SFT training script with memory-efficient settings
echo 'Starting optimized SFT training...';
python llada_sft_optimized.py \
  --epochs 3 \
  --batch-size 1 \
  --lr 1e-5 \
  --grad-accum 16 \
  --max-length 384 \
  --checkpoint-dir checkpoints \
  --save-steps 50
"
