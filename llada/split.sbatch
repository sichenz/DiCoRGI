#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --mem=32GB
#SBATCH --job-name=split
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=split.out

module purge

singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/$USER/DiCoRGI/llada; python data_split.py --input_dir "/scratch/$USER/DiCoRGI/data_gen/gen10000/tasks" --output_dir "/scratch/$USER/DiCoRGI/llada/llada_data" --training_count 9000 --testing_count 1000 --num_workers 2"


