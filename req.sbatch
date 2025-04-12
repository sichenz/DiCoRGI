#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00  
#SBATCH --mem=32GB
#SBATCH --job-name=req
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=req.out

module purge

singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; \
  pip install -r /scratch/$USER/DiCoRGI/requirements.txt"
