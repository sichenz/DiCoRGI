import torch
import numpy as np
import torch.nn.functional as F
import json
import random
import multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import gc
from contextlib import contextmanager

# Distributed training utilities
def setup_distributed():
    """Initialize distributed training."""
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        return True
    return False

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

@contextmanager
def gpu_context(gpu_id):
    """Context manager for GPU operations."""
    try:
        torch.cuda.set_device(gpu_id)
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# SFT Dataset class (same as before)
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length or 2048
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        formatted_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
        
        formatted_text = self.tokenizer.apply_chat_template(
            formatted_messages, 
            add_generation_prompt=False, 
            tokenize=False
        )
        
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        user_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], 
            add_generation_prompt=True, 
            tokenize=False
        )
        prompt_tokens = self.tokenizer(user_prompt, add_special_tokens=False)['input_ids']
        prompt_length = len(prompt_tokens)
        
        return {
            'input_ids': tokenized['input_ids'],
            'prompt_length': prompt_length,
            'full_length': len(tokenized['input_ids'])
        }

def collate_sft_batch(batch):
    max_length = max(item['full_length'] for item in batch)
    input_ids = []
    prompt_lengths = []
    
    for item in batch:
        padded_ids = item['input_ids'] + [126337] * (max_length - item['full_length'])
        input_ids.append(padded_ids)
        prompt_lengths.append(item['prompt_length'])
    
    return {
        'input_ids': torch.tensor(input_ids),
        'prompt_lengths': torch.tensor(prompt_lengths)
    }

# Training functions (same as before but with distributed support)
def train_sft_distributed(model, tokenizer, sft_data, epochs=10, batch_size=4, learning_rate=1e-5):
    """Train with distributed data parallel support."""
    is_distributed = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{rank}")
    
    if is_distributed:
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
    
    dataset = SFTDataset(sft_data, tokenizer)
    
    # Create distributed sampler
    if is_distributed:
        sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_sft_batch,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
    
    for epoch in range(epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            loss = sft_training_step(model, batch, optimizer)
            total_loss += loss
            
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    
    cleanup_distributed()
    return model

# Parallel testing functions
def process_task_batch(args):
    """Process a batch of tasks on a specific GPU."""
    batch_tasks, gpu_id, model_path, tokenizer_path = args
    
    try:
        with gpu_context(gpu_id):
            device = f"cuda:{gpu_id}"
            
            # Load model and tokenizer for this GPU
            model = AutoModel.from_pretrained(model_path, 
                                              trust_remote_code=True, 
                                              torch_dtype=torch.bfloat16, 
                                              use_cache=False, 
                                              device_map=device).eval()
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
            results = []
            for task_id, test_idx, test_item, correct_output, prompt_text in batch_tasks:
                result = process_single_test(model, tokenizer, task_id, test_idx, 
                                             test_item, correct_output, prompt_text, device)
                results.append(result)
                
                # Clear cache periodically
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            return results
    
    except Exception as e:
        print(f"Error in batch processing on GPU {gpu_id}: {e}")
        return []
    
    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def process_single_test(model, tokenizer, task_id, test_idx, test_item, 
                        correct_output, prompt_text, device):
    """Process a single test case."""
    try:
        test_input_grid = json.dumps(test_item['input'])
        full_prompt = prompt_text + f"Test Input:\n{test_input_grid}\n\nWhat should be the Output grid? Only provide the output grid as your answer."
        
        m = [{"role": "user", "content": full_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        
        input_ids = tokenizer(formatted_prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # Generate output
        out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, 
                       temperature=0., cfg_scale=0., remasking='low_confidence')
        
        # Decode the output
        generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Extract generated grid
        generated_grid = extract_grid_from_text(generated_text)
        
        # Calculate accuracy
        accuracy = compare_grids(generated_grid, correct_output)
        
        return {
            'task_id': task_id,
            'test_idx': test_idx,
            'accuracy': accuracy,
            'generated_grid': generated_grid,
            'correct_grid': correct_output,
            'generated_text': generated_text,
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'task_id': task_id,
            'test_idx': test_idx,
            'accuracy': 0.0,
            'error': str(e),
            'status': 'error'
        }

def test_model_on_all_problems_parallel(model_path, tokenizer_path, arc_data, arc_solutions, n_gpus=2):
    """Test the model on all problems using multiple GPUs."""
    # Prepare all test tasks
    all_tasks = []
    
    for task_id, task_data in arc_data.items():
        train_examples = task_data['train']
        test_examples = task_data['test']
        
        # Build prompt text once per task
        prompt_text = """You are given a set of input-output grid pairs that define a transformation rule. Each grid is a 2D array of integers, where each integer represents a color. Your task is to learn the transformation and apply it to a new input grid.

Each example below consists of an 'input' grid and its corresponding 'output' grid. Analyze the patterns and apply the inferred transformation rule to the final test input.

You must determine the correct size of the output grid based on the transformation logic. However, the output grid must not exceed 30 rows or 30 columns in size.

Training Examples:
"""
        for idx, train_item in enumerate(train_examples):
            input_grid = json.dumps(train_item['input'])
            output_grid = json.dumps(train_item['output'])
            prompt_text += f"Input: {input_grid}\nOutput: {output_grid}\n\n"
        
        for test_idx, test_item in enumerate(test_examples):
            if task_id in arc_solutions and test_idx < len(arc_solutions[task_id]):
                correct_output = arc_solutions[task_id][test_idx]
                all_tasks.append((task_id, test_idx, test_item, correct_output, prompt_text))
    
    # Split tasks into batches for each GPU
    batch_size = max(1, len(all_tasks) // (n_gpus * 4))  # ~4 batches per GPU
    task_batches = []
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        gpu_id = (i // batch_size) % n_gpus
        task_batches.append((batch, gpu_id, model_path, tokenizer_path))
    
    # Process batches in parallel
    print(f"Processing {len(all_tasks)} tests in {len(task_batches)} batches using {n_gpus} GPUs")
    
    with mp.Pool(processes=n_gpus) as pool:
        batch_results = []
        for results in tqdm(pool.imap(process_task_batch, task_batches), total=len(task_batches), desc="Processing batches"):
            batch_results.extend(results)
    
    # Combine and organize results
    organized_results = {}
    for result in batch_results:
        task_id = result['task_id']
        if task_id not in organized_results:
            organized_results[task_id] = []
        organized_results[task_id].append(result)
    
    # Print detailed results
    total_accuracy = 0.0
    total_problems = 0
    overall_stats = []
    
    for task_id in sorted(organized_results.keys()):
        task_results = organized_results[task_id]
        print("\n" + "="*50)
        print(f"TASK ID: {task_id}")
        print("="*50)
        
        for result in task_results:
            if result['status'] == 'success':
                print(f"\nTest Example {result['test_idx']}")
                
                if result['correct_grid']:
                    print(f"Correct Output Shape: {len(result['correct_grid'])}x{len(result['correct_grid'][0])}")
                else:
                    print("Correct Output Shape: Unknown")
                    
                if result['generated_grid']:
                    print(f"Generated Output Shape: {len(result['generated_grid'])}x{len(result['generated_grid'][0])}")
                else:
                    print("Generated Output Shape: Could not extract grid")
                
                print(f"ACCURACY: {result['accuracy']:.2f}%")
                
                # Display side by side comparison
                if result['correct_grid'] and result['generated_grid']:
                    display_grids_side_by_side(result['correct_grid'], result['generated_grid'])
                
                total_accuracy += result['accuracy']
                total_problems += 1
            else:
                print(f"\nTest Example {result['test_idx']}: ERROR - {result.get('error', 'Unknown error')}")
            
            overall_stats.append(result)
    
    # Print overall summary
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    print(f"Total problems processed: {total_problems}")
    print(f"Average accuracy: {total_accuracy/total_problems:.2f}%")
    print(f"Perfect accuracy problems: {len([s for s in overall_stats if s.get('accuracy', 0) == 100.0])}")
    
    return overall_stats

# Include all other utility functions from the original code
# (forward_process, sft_training_step, extract_grid_from_text, compare_grids, 
#  display_grids_side_by_side, create_arc_sft_data, generate functions)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLaDA model on ARC dataset with parallel processing')
    parser.add_argument('--skip-training', action='store_true', help='Skip fine-tuning and only test')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n-gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--n-cpus', type=int, default=8, help='Number of CPUs to use')
    
    args = parser.parse_args()
    
    # Set CPU and GPU counts
    mp.set_start_method('spawn', force=True)
    
    # Create output directory for results
    output_dir = 'sft_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ARC data
    print("Loading ARC data...")
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_challenges.json', 'r') as f:
        arc_data = json.load(f)
    
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_solutions.json', 'r') as f:
        arc_solutions = json.load(f)
    
    if not args.skip_training:
        # Load model and tokenizer for training
        print("Loading model and tokenizer for training...")
        model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', 
                                          trust_remote_code=True, 
                                          torch_dtype=torch.bfloat16, 
                                          use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        
        # Create SFT data
        print("Creating SFT data...")
        sft_data = create_arc_sft_data(arc_data, arc_solutions)
        print(f"Created {len(sft_data)} SFT training examples")
        
        # Train the model with distributed training
        print("Starting distributed SFT fine-tuning...")
        model = train_sft_distributed(model, tokenizer, sft_data, 
                                      epochs=args.epochs, 
                                      batch_size=args.batch_size, 
                                      learning_rate=args.lr)
        print("SFT fine-tuning completed!")
        
        # Save the fine-tuned model
        save_path = f'sft_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f"Saving model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        model_path = save_path
        tokenizer_path = save_path
    else:
        # Use the original model for testing
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'
        tokenizer_path = 'GSAI-ML/LLaDA-8B-Instruct'
    
    # Test the model on all problems in parallel
    print("\nTesting model on all problems with parallel processing...")
    overall_stats = test_model_on_all_problems_parallel(model_path, tokenizer_path, 
                                                        arc_data, arc_solutions, 
                                                        n_gpus=args.n_gpus)
    
    # Save results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/test_results.json")

if __name__ == '__main__':
    main()