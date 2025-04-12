#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=llada
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=llada.out
#SBATCH --gres=gpu:1

module purge

# Ensure CUDA is properly detected
export CUDA_VISIBLE_DEVICES=0
echo '=== GPU Information ==='
nvidia-smi
nvcc --version 2>/dev/null || echo "nvcc not available (this is normal)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/$USER/DiCoRGI/llada; python llada_arc_optimized.py \
    --train_data_dir /scratch/sz4972/DiCoRGI/llada/llada_data/training \
    --test_data_dir /scratch/sz4972/DiCoRGI/llada/llada_data/testing \
    --output_dir /scratch/sz4972/DiCoRGI/llada/llada_arc_models \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --seed 42 \
    --num_workers 4 \
    --eval_steps 50 \
    --eval_temperature 0.5 \
    --mixed_precision \
    --gradient_checkpointing \
    --memory_efficient \
    --files_per_subset 5 \
    --clean_memory_steps 250 \
    --max_examples_per_file 100"

