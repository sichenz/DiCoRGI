#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=llada_sft
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=llada_sft.out

# Create results directory
export RESULTS_DIR=llada_sft_results
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"

# Create a directory for the implementation
mkdir -p $RESULTS_DIR/implementation
cd $RESULTS_DIR/implementation

# Download the diffu_grpo_trainer.py file directly from GitHub
echo "Downloading implementation file..."
wget -q https://raw.githubusercontent.com/dllm-reasoning/d1/main/diffu-grpo/diffu_grpo_trainer.py
echo "Download complete"

# Dataset and problem specific settings
KEY="007bbfb7"
ARC_DATA_PATH="/scratch/sz4972/diffuseARC/llada/experiments/arc_data/arc-agi_training_challenges.json"
MODEL_PATH="LLaDA-sft-s1k"
NUM_ITER=8
TEMP=0.3

# Return to main directory
cd ../..

# Run our Python script that imports DiffuGRPOTrainer
echo "Starting ARC solver..."
srun --output $RESULTS_DIR/arc_%j.out \
    python llada_sft.py \
        --model_path ${MODEL_PATH} \
        --data_path ${ARC_DATA_PATH} \
        --key ${KEY} \
        --output_dir ${RESULTS_DIR} \
        --num_iterations ${NUM_ITER} \
        --temperature ${TEMP}