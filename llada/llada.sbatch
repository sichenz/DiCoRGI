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

singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/$USER/DiCoRGI/llada; python llada_arc.py \
    --train_data_dir /scratch/sz4972/DiCoRGI/llada/llada_data/training \
    --test_data_dir /scratch/sz4972/DiCoRGI/llada/llada_data/testing \
    --output_dir /scratch/sz4972/DiCoRGI/llada/llada_arc_models \
    --d_model 768 \
    --nhead 12 \
    --num_layers 12 \
    --dim_feedforward 3072 \
    --dropout 0.1 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --seed 42 \
    --num_workers 4 \
    --eval_steps 50 \
    --eval_temperature 0.5"

