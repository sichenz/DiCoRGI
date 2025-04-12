#!/usr/bin/env python3
import os
import json
import random
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_file(file_path, output_base_dir, training_count, testing_count, seed=42):
    """
    Process a single JSON file and split its examples into training and testing sets.
    
    Args:
        file_path (Path): Path to the JSON file
        output_base_dir (Path): Base directory for output
        training_count (int): Number of examples for training
        testing_count (int): Number of examples for testing
        seed (int): Random seed for reproducibility
    """
    # Create output directories if they don't exist
    training_dir = output_base_dir / "training"
    testing_dir = output_base_dir / "testing"
    training_dir.mkdir(exist_ok=True, parents=True)
    testing_dir.mkdir(exist_ok=True, parents=True)
    
    # Get filename without extension
    filename = file_path.stem
    
    try:
        # Load data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if we have enough examples
        total_examples = len(data)
        if total_examples < (training_count + testing_count):
            print(f"Warning: {file_path} has only {total_examples} examples, " 
                  f"which is less than requested {training_count + testing_count}. "
                  f"Using all available examples with same ratio.")
            
            # Adjust counts proportionally
            ratio = total_examples / (training_count + testing_count)
            training_count = int(training_count * ratio)
            testing_count = total_examples - training_count
        
        # Shuffle the data with a fixed seed for reproducibility
        random.seed(seed)
        random.shuffle(data)
        
        # Split the data
        training_data = data[:training_count]
        testing_data = data[training_count:training_count + testing_count]
        
        # Save training data
        with open(training_dir / f"{filename}.json", 'w') as f:
            json.dump(training_data, f)
        
        # Save testing data
        with open(testing_dir / f"{filename}.json", 'w') as f:
            json.dump(testing_data, f)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def split_arc_data(input_dir, output_dir, training_count=9000, testing_count=1000, num_workers=None, seed=42):
    """
    Split ARC task data into training and testing sets.
    
    Args:
        input_dir (str): Directory containing JSON files with ARC task examples
        output_dir (str): Directory to save split data
        training_count (int): Number of examples for training set
        testing_count (int): Number of examples for testing set
        num_workers (int): Number of worker processes to use (None means use all available)
        seed (int): Random seed for reproducibility
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all JSON files in the input directory
    json_files = list(input_path.glob('*.json'))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {total_files} JSON files in {input_dir}")
    print(f"Splitting each file into {training_count} training and {testing_count} testing examples")
    
    # Set up parallelization
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    num_workers = min(num_workers, total_files)  # Don't use more workers than files
    print(f"Using {num_workers} worker processes")
    
    # Set up progress bar
    pbar = tqdm(total=total_files, desc="Processing files")
    
    # Function to update progress bar
    def update_progress(result):
        pbar.update(1)
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_file, 
        output_base_dir=output_path, 
        training_count=training_count, 
        testing_count=testing_count,
        seed=seed
    )
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file_path in json_files:
            future = executor.submit(process_func, file_path)
            future.add_done_callback(lambda p: update_progress(p.result()))
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    pbar.close()
    print(f"Done! Processed {total_files} files.")
    print(f"Training data saved to: {output_path / 'training'}")
    print(f"Testing data saved to: {output_path / 'testing'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split ARC task data into training and testing sets')
    parser.add_argument('--input_dir', type=str, default='DiCoRGI/data_gen/gen10000/tasks',
                        help='Directory containing JSON files with ARC task examples')
    parser.add_argument('--output_dir', type=str, default='DiCoRGI/llada/llada_data',
                        help='Directory to save split data')
    parser.add_argument('--training_count', type=int, default=9000,
                        help='Number of examples for training set')
    parser.add_argument('--testing_count', type=int, default=1000,
                        help='Number of examples for testing set')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes to use (default: all available CPUs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    split_arc_data(
        args.input_dir,
        args.output_dir,
        args.training_count,
        args.testing_count,
        args.num_workers,
        args.seed
    )