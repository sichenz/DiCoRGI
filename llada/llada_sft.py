import sys
import json
import torch
import argparse
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
import numpy as np

# Add the directory with the downloaded file to the path
sys.path.append('./d1_implementation')

# Import the DiffuGRPOTrainer from the downloaded file
from diffu_grpo_trainer import DiffuGRPOTrainer

def load_arc_data(data_path, key):
    """Load ARC data for a specific key"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if key not in data:
        raise ValueError(f"Key {key} not found in dataset")
    
    return data[key]

def format_arc_prompt(arc_data):
    """Format ARC data into a prompt for the model"""
    prompt = "Solve the following Abstract Reasoning Challenge (ARC):\n\n"
    
    # Add training examples
    prompt += "Training examples:\n"
    for idx, example in enumerate(arc_data["train"]):
        prompt += f"Example {idx+1}:\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"
        prompt += f"Output:\n{json.dumps(example['output'])}\n\n"
    
    # Add test example
    prompt += "Now solve this example:\n"
    prompt += f"Input:\n{json.dumps(arc_data['test'][0]['input'])}\n"
    prompt += "Output:\n"
    
    return prompt

def arc_reward_function(prompts, completions, **kwargs):
    """Reward function for ARC problems"""
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        try:
            # Parse the completion to extract the predicted output matrix
            import re
            matrix_match = re.search(r'\[\s*\[.*?\]\s*\]', completion, re.DOTALL)
            
            if matrix_match:
                predicted_output_str = matrix_match.group(0)
                # Clean up the string and parse as JSON
                predicted_output_str = predicted_output_str.replace('\n', '')
                predicted_output = json.loads(predicted_output_str)
                
                # For this simple implementation, we'll just check if the output
                # has the correct structure (a 2D matrix)
                if isinstance(predicted_output, list) and all(isinstance(row, list) for row in predicted_output):
                    # In a real implementation, you would check against expected patterns
                    # or verify the transformation logic
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0  # No valid matrix found
        except Exception as e:
            print(f"Error evaluating completion: {e}")
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="LLaDA-sft-s1k-merged")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/arc_agi")
    parser.add_argument("--num_iterations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()
    
    # Load the model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load ARC data for specific key
    print(f"Loading ARC data for key {args.key}")
    arc_data = load_arc_data(args.data_path, args.key)
    
    # Format the prompt
    formatted_prompt = format_arc_prompt(arc_data)
    print(f"Formatted prompt: {formatted_prompt[:200]}...")
    
    # Create a dataset with the formatted prompt
    dataset = Dataset.from_dict({
        "prompt": [{"prompt": formatted_prompt}],
        "key": [args.key]
    })
    
    # Configure the trainer
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        temperature=args.temperature,
        diffusion_steps=128,
        block_length=128,
        mask_id=126336,  # Mask token ID for LLaDA
        p_mask_prompt=0.5,
        cfg_scale=0.0,
        remasking="low_confidence",
        random_masking=True,
    )
    
    print("Creating DiffuGRPOTrainer")
    # Create and run the trainer
    trainer = DiffuGRPOTrainer(
        model=model,
        reward_funcs=[arc_reward_function],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting training")
    # Train the model
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    # Save the trained model
    trainer.save_model()
    
    print("Done!")

if __name__ == "__main__":
    main()