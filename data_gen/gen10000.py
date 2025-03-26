#!/usr/bin/env python3
import os
import sys
import json
import time
import multiprocessing as mp
from functools import partial
import shutil
import glob
import tqdm

# Import the original code
from main import *

def generate_subset(keys, subset_path, seed, n_examples, diff_lb=0, diff_ub=1):
    """Generate dataset for a subset of tasks"""
    # Create a subset directory
    os.makedirs(subset_path, exist_ok=True)
    tasks_path = os.path.join(subset_path, 'tasks')
    os.makedirs(tasks_path, exist_ok=True)
    
    # Get the generator and verifier functions
    generators_mapper = get_generators()
    verifiers_mapper = get_verifiers()
    
    # Process only the assigned subset of keys
    metadata = {}
    
    # Create a progress bar for tasks assigned to this process
    task_pbar = tqdm.tqdm(keys, desc=f"Process {os.getpid()}", 
                          position=mp.current_process()._identity[0] % 5, 
                          leave=True)
    
    for key in task_pbar:
        generator = generators_mapper[key]
        verifier = verifiers_mapper[key]
        
        # Update progress bar description to show current task
        task_pbar.set_description(f"P{os.getpid()} | Task {key}")
        
        # Create a temporary file to indicate this task is being processed
        with open(os.path.join(tasks_path, f'{key}.in_progress'), 'w') as f:
            f.write(str(os.getpid()))
        
        # Generate examples and collect stats
        seen = set()
        examples = []
        stats = {
            'n_generations': 0, 'n_verified': 0, 'n_nondegenerate': 0,
            'rng_difficulties': [], 'pso_difficulties': []
        }
        start = time.time()
        
        # Create a nested progress bar for examples within this task
        example_pbar = tqdm.tqdm(total=n_examples, 
                                 desc=f"Examples for {key}",
                                 position=mp.current_process()._identity[0] % 5 + 5,
                                 leave=False)
        
        while len(examples) < n_examples:
            example, identifier, success = None, None, True
            try:
                example = generator(diff_lb, diff_ub)
                assert is_grid(example['input'])
                assert is_grid(example['output'])
                identifier = hash(example['input'])
                stats['n_generations'] += 1
            except:
                success = False
            try:
                assert success and verifier(example['input']) == example['output']
                stats['n_verified'] += 1
            except:
                success = False
            try:
                assert success and example['input'] != example['output']
                stats['n_nondegenerate'] += 1
            except:
                success = False
            if success and identifier not in seen:
                examples.append(example)
                seen.add(identifier)
                stats['rng_difficulties'].append(get_rng_difficulty(example))
                stats['pso_difficulties'].append(get_pso_difficulty(example))
                example_pbar.update(1)  # Update the inner progress bar
                
                # Update outer progress bar postfix to show progress
                task_pbar.set_postfix(examples=f"{len(examples)}/{n_examples}")
        
        # Close the examples progress bar
        example_pbar.close()
        
        end = time.time()
        stats['runtime'] = end - start
        
        # Save examples and stats
        with open(os.path.join(tasks_path, f'{key}.json'), 'w') as fp:
            json.dump(examples, fp)
        
        metadata[key] = stats
        
        # Remove the in-progress file
        os.remove(os.path.join(tasks_path, f'{key}.in_progress'))
    
    # Close the tasks progress bar
    task_pbar.close()
    
    # Save metadata for this subset
    with open(os.path.join(subset_path, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)
    
    return subset_path

def merge_datasets(final_path, subset_paths):
    """Merge multiple subset datasets into one final dataset"""
    # Create final directory
    os.makedirs(final_path, exist_ok=True)
    tasks_path = os.path.join(final_path, 'tasks')
    os.makedirs(tasks_path, exist_ok=True)
    
    print("Merging results from all processes...")
    
    # Combine metadata
    combined_metadata = {}
    for subset_path in subset_paths:
        with open(os.path.join(subset_path, 'metadata.json'), 'r') as fp:
            metadata = json.load(fp)
        combined_metadata.update(metadata)
    
    # Save combined metadata
    with open(os.path.join(final_path, 'metadata.json'), 'w') as fp:
        json.dump(combined_metadata, fp)
    
    # Copy all task files with a progress bar
    all_files = []
    for subset_path in subset_paths:
        subset_tasks_path = os.path.join(subset_path, 'tasks')
        files = glob.glob(os.path.join(subset_tasks_path, '*.json'))
        all_files.extend(files)
    
    with tqdm.tqdm(all_files, desc="Copying task files") as pbar:
        for file_path in pbar:
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(tasks_path, file_name))
    
    print(f"All subsets merged into {final_path}")

def parallelize_dataset_generation(path='gen10000', seed=42, n_examples=1000, num_workers=None):
    """Generate a dataset in parallel without modifying main.py"""
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Get all task keys
    generators_mapper = get_generators()
    keys = sorted(generators_mapper.keys())
    print(f"Found {len(keys)} tasks to process")
    
    # Split keys into chunks for each worker
    chunks = [[] for _ in range(num_workers)]
    for i, key in enumerate(keys):
        chunks[i % num_workers].append(key)
    
    # Create temporary directories for each subset
    temp_dir = f"{path}_temp"
    os.makedirs(temp_dir, exist_ok=True)
    subset_paths = [os.path.join(temp_dir, f"subset_{i}") for i in range(num_workers)]
    
    # Print task distribution
    for i, chunk in enumerate(chunks):
        print(f"Process {i} will handle {len(chunk)} tasks")
    
    # Configure tqdm to work with multiple processes
    mp.set_start_method('spawn', force=True)
    
    # Create a process pool
    pool = mp.Pool(processes=num_workers)
    
    print("Starting parallel processing...")
    
    # Submit tasks to the pool
    results = []
    for i, chunk in enumerate(chunks):
        subset_seed = seed + i  # Derive unique seed for each process
        subset_path = subset_paths[i]
        results.append(pool.apply_async(
            generate_subset, 
            args=(chunk, subset_path, subset_seed, n_examples)
        ))
    
    # Wait for all processes to complete
    pool.close()
    pool.join()
    
    # Collect results
    completed_paths = [r.get() for r in results]
    
    # Merge results
    merge_datasets(path, completed_paths)
    
    # Clean up temp directories
    shutil.rmtree(temp_dir)
    
    print(f"Dataset generation complete. Results saved to {path}")

if __name__ == "__main__":
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = mp.cpu_count()
    
    parallelize_dataset_generation(
        path='gen10000',
        seed=0,
        n_examples=10000,
        num_workers=num_workers
    )
    
# generate_dataset(path='gen10000', seed=0, n_examples=10000)
