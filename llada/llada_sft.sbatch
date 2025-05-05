#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=128GB
#SBATCH --job-name=llada_sft
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llada_sft.out

echo "Running on host: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Current working directory: $(pwd)"

# Module purge to clean the environment
module purge

singularity exec --nv \
--overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
/bin/bash -c "
source /ext3/env.sh; 
conda activate arc; 
cd /scratch/sz4972/DiCoRGI/llada;
echo 'Python version:';
python --version;
echo 'CUDA available:';
python -c 'import torch; print(torch.cuda.is_available())';
echo 'Starting SFT training...';
python llada_sft.py --epochs 5 --batch-size 1 --lr 1e-5
"

