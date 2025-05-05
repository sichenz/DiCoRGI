import torch
import numpy as np
import torch.nn.functional as F
import json
import argparse
import multiprocessing as mp
from tqdm import tqdm
import os
import sys
import gc
import time
from contextlib import contextmanager

from transformers import AutoModel, AutoTokenizer

# generate.py functions
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

def extract_grid_from_text(text):
    """Extract a grid from the generated text."""
    try:
        import re

        # Look for grid patterns in the text
        # First try to find a proper JSON array
        match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
        if match:
            grid_str = match.group(0)
            return json.loads(grid_str)

        # If not found, try to extract line by line
        lines = text.strip().split('\n')
        grid = []
        for line in lines:
            # Look for lines that look like arrays
            match = re.search(r'\[.*?\]', line)
            if match:
                try:
                    row = json.loads(match.group(0))
                    if isinstance(row, list) and all(isinstance(x, int) for x in row):
                        grid.append(row)
                except:
                    continue

        if grid:
            return grid

        # Last resort: look for numbers
        grid = []
        for line in lines:
            numbers = re.findall(r'\d+', line)
            if numbers:
                row = [int(num) for num in numbers]
                grid.append(row)

        return grid if grid else None
    except Exception as e:
        print(f"Error extracting grid: {e}")
        return None

def compare_grids(grid1, grid2):
    """Compare two grids and return accuracy percentage."""
    if grid1 is None or grid2 is None:
        return 0.0

    # Ensure both grids are 2D lists
    if not isinstance(grid1, list) or not isinstance(grid2, list):
        return 0.0

    # Check if dimensions match
    if len(grid1) != len(grid2):
        return 0.0

    total_cells = 0
    matching_cells = 0

    for row1, row2 in zip(grid1, grid2):
        if not isinstance(row1, list) or not isinstance(row2, list):
            continue
        if len(row1) != len(row2):
            continue

        for cell1, cell2 in zip(row1, row2):
            total_cells += 1
            if cell1 == cell2:
                matching_cells += 1

    if total_cells == 0:
        return 0.0

    return (matching_cells / total_cells) * 100

def visualize_grids_side_by_side(grid1, grid2, title1="Correct Output", title2="Generated Grid"):
    """Display two grids side by side with color coding."""
    if not grid1 or not grid2:
        print("One or both grids are empty")
        return

    max_height = max(len(grid1), len(grid2)) if grid1 and grid2 else 0

    # Add padding to shorter grid
    padded_grid1 = grid1 + [[0] * len(grid1[0])] * (max_height - len(grid1)) if len(grid1) < max_height else grid1
    padded_grid2 = grid2 + [[0] * len(grid2[0])] * (max_height - len(grid2)) if len(grid2) < max_height else grid2

    print(f"\n{title1:<30} | {title2}")
    print("-" * 30 + " | " + "-" * 30)

    for i in range(max_height):
        if i < len(grid1):
            row1_str = " ".join(f"{cell:2d}" for cell in grid1[i])
        else:
            row1_str = "  " * len(grid2[0]) if grid2 else ""

        if i < len(grid2):
            row2_str = " ".join(f"{cell:2d}" for cell in grid2[i])
        else:
            row2_str = "  " * len(grid2[0]) if grid2 else ""

        print(f"{row1_str:<30} | {row2_str}")

def display_grid_horizontally(grid, title):
    """Display a grid in a horizontal format."""
    if not grid:
        print(f"{title}: Empty grid")
        return

    print(f"\n{title}:")
    for row in grid:
        print(" ".join(map(str, row)))

@contextmanager
def gpu_context(gpu_id):
    """Context manager for GPU operations."""
    try:
        torch.cuda.set_device(gpu_id)
        yield
    finally:
        torch.cuda.empty_cache()

def print_task_results(task_id, correct_output, generated_grid, accuracy):
    """Print detailed results for a single task."""
    print("\n" + "="*80)
    print(f"Task ID: {task_id}")
    
    # Print shapes
    if correct_output:
        correct_shape = f"{len(correct_output)}x{len(correct_output[0])}"
    else:
        correct_shape = "N/A"
    
    if generated_grid:
        generated_shape = f"{len(generated_grid)}x{len(generated_grid[0])}"
    else:
        generated_shape = "Could not extract grid"
    
    print(f"Correct Output Shape: {correct_shape}")
    print(f"Generated Output Shape: {generated_shape}")
    
    # Print grids
    display_grid_horizontally(correct_output, "Correct Output")
    display_grid_horizontally(generated_grid, "Generated Grid")
    
    # Print accuracy
    print(f"\nACCURACY: {accuracy:.2f}%")
    
    # Print side by side comparison
    visualize_grids_side_by_side(correct_output, generated_grid)
    print("="*80)

def process_batch(args):
    """Process a batch of tasks."""
    batch_tasks, gpu_id, solutions = args
    
    results = []
    model = None
    tokenizer = None
    
    try:
        with gpu_context(gpu_id):
            device = f"cuda:{gpu_id}"
            
            # Load model once for the entire batch
            model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', 
                                              trust_remote_code=True, 
                                              torch_dtype=torch.bfloat16, 
                                              use_cache=False, 
                                              device_map=device).eval()
            tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
            
            for task_id, task_data in batch_tasks:
                try:
                    result = process_single_task(model, tokenizer, task_id, task_data, device, solutions)
                    results.append(result)
                    
                    # Print detailed results immediately after processing each task
                    print_task_results(
                        task_id, 
                        result['correct_grid'], 
                        result['generated_grid'], 
                        result['accuracy']
                    )
                    
                except Exception as e:
                    results.append({
                        'task_id': task_id,
                        'accuracy': 0.0,
                        'error': str(e),
                        'status': 'error'
                    })
                    print(f"\nERROR processing task {task_id}: {e}")
                
                # Periodically clear cache
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
    
    except Exception as e:
        print(f"Error in batch processing for GPU {gpu_id}: {e}")
        # Fill remaining tasks with errors
        for task_id, _ in batch_tasks[len(results):]:
            results.append({
                'task_id': task_id,
                'accuracy': 0.0,
                'error': f"Batch processing error: {e}",
                'status': 'error'
            })
    
    finally:
        # Clean up
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

def process_single_task(model, tokenizer, task_id, task_data, device, solutions):
    """Process a single task."""
    train = task_data['train']
    test = task_data['test']
    test_input = test[0]['input']

    # Build training examples
    examples = ""
    for item in train:
        input_grid = json.dumps(item['input'])
        output_grid = json.dumps(item['output'])
        examples += f"Input: {input_grid}\nOutput: {output_grid}\n\n"

    # Create the prompt for the LLaDA model
    prompt = f"""You are given a set of input-output grid pairs that define a transformation rule. Each grid is a 2D array of integers, where each integer represents a color. Your task is to learn the transformation and apply it to a new input grid.

Each example below consists of an 'input' grid and its corresponding 'output' grid. Analyze the patterns and apply the inferred transformation rule to the final test input.

You must determine the correct size of the output grid based on the transformation logic. However, the output grid must not exceed 30 rows or 30 columns in size.

Training Examples:
{examples}

Test Input:
{json.dumps(test_input)}

What should be the Output grid? Only provide the output grid as your answer."""

    # Add special tokens for the Instruct model
    m = [{"role": "user", "content": prompt}, ]
    formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    # Tokenize and generate
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Generate output
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, 
                   temperature=0., cfg_scale=0., remasking='low_confidence')

    # Decode the output
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Extract generated grid
    generated_grid = extract_grid_from_text(generated_text)

    # Get correct output
    correct_output = solutions[task_id][0]

    # Calculate accuracy
    accuracy = compare_grids(generated_grid, correct_output)

    return {
        'task_id': task_id,
        'accuracy': accuracy,
        'generated_grid': generated_grid,
        'correct_grid': correct_output,
        'generated_text': generated_text,
        'status': 'success'
    }

def main():
    parser = argparse.ArgumentParser(description='Run LLaDA on ARC problems in parallel')
    parser.add_argument('num_devices', type=int, help='Number of GPUs/CPUs to use')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size per GPU')
    args = parser.parse_args()

    # Set number of devices to use
    n_devices = args.num_devices
    batch_size = args.batch_size

    # Create output directory
    output_dir = 'llada_base_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load data from JSON file
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_challenges.json', 'r') as f:
        data = json.load(f)

    # Load correct solutions
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    # Prepare tasks for parallel processing
    task_ids = list(data.keys())
    all_tasks = [(task_id, data[task_id]) for task_id in task_ids]
    
    # Split tasks into batches
    batches = []
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        gpu_id = (i // batch_size) % n_devices
        batches.append((batch, gpu_id, solutions))

    # Process batches in parallel
    print(f"Processing {len(all_tasks)} tasks in {len(batches)} batches using {n_devices} GPUs")
    
    start_time = time.time()
    
    with mp.Pool(processes=n_devices) as pool:
        all_results = []
        for batch_results in tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing batches"):
            all_results.extend(batch_results)

    processing_time = time.time() - start_time

    # Calculate overall statistics
    successful_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] == 'error']

    total_tasks = len(all_results)
    total_accuracy = sum(r['accuracy'] for r in successful_results)
    avg_accuracy = total_accuracy / len(successful_results) if successful_results else 0.0

    # Save results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    summary = {
        'total_tasks': total_tasks,
        'successful_tasks': len(successful_results),
        'failed_tasks': len(failed_results),
        'average_accuracy': avg_accuracy,
        'perfect_accuracy_tasks': len([r for r in successful_results if r['accuracy'] == 100.0]),
        'processing_time_seconds': processing_time,
        'tasks_per_second': total_tasks / processing_time if processing_time > 0 else 0
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save errors
    if failed_results:
        with open(os.path.join(output_dir, 'error_log.json'), 'w') as f:
            json.dump(failed_results, f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {len(successful_results)}")
    print(f"Failed tasks: {len(failed_results)}")
    print(f"Average accuracy: {avg_accuracy:.2f}%")
    print(f"Perfect accuracy tasks: {len([r for r in successful_results if r['accuracy'] == 100.0])}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Tasks per second: {total_tasks/processing_time:.2f}")
    print(f"\nDetailed results saved to {output_dir}/")

if __name__ == '__main__':
    main()