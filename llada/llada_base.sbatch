#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=64GB
#SBATCH --job-name=llada_base
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llada_base.out

singularity exec --nv \
--overlay /scratch/sz4972/jupyter_env/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
/bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/sz4972/DiCoRGI/llada; python llada_base.py"