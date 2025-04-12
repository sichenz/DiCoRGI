#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00  # Increased time to allow for package installation
#SBATCH --mem=32GB
#SBATCH --job-name=split
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=split.out

module purge

singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; \
  conda install transformers -c huggingface"

