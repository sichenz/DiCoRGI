#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=llada
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llada.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER/DiCoRGI/llada

# Then run the notebook
singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/$USER/DiCoRGI/llada; python run.py"
"   
