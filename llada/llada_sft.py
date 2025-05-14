import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import gc
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Parse command line arguments
parser = argparse.ArgumentParser(description='LLaDA SFT Fine-tuning')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--grad-accum', type=int, default=16, help='Gradient accumulation steps')
parser.add_argument('--max-length', type=int, default=384, help='Maximum sequence length')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--save-steps', type=int, default=100, help='Save checkpoint every X steps')
args = parser.parse_args()

# Constants
MASK_ID = 126336
EOS_TOKEN_ID = 128001  # Adjust based on your tokenizer

class ARCDataset(Dataset):
    def __init__(self, data, solutions, tokenizer, max_length=384):
        self.data = data
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._prepare_data()
    
    def _prepare_data(self):
        samples = []
        for task_id, task_data in self.data.items():
            if task_id not in self.solutions:
                continue
            
            train = task_data['train']
            test = task_data['test']
            
            # Combine all training examples
            examples = ""
            for item in train:
                input_grid = json.dumps(item['input'])
                output_grid = json.dumps(item['output'])
                examples += f"Input: {input_grid}\nOutput: {output_grid}\n\n"
            
            # Create prompt for each test case
            for i, test_item in enumerate(test):
                test_input = test_item['input']
                correct_output = self.solutions[task_id][i]
                
                prompt = f"""You are given a set of input-output grid pairs that define a transformation rule. Each grid is a 2D array of integers, where each integer represents a color. Your task is to learn the transformation and apply it to a new input grid.

Each example below consists of an 'input' grid and its corresponding 'output' grid. Analyze the patterns and apply the inferred transformation rule to the final test input.

You must determine the correct size of the output grid based on the transformation logic. However, the output grid must not exceed 30 rows or 30 columns in size.

Training Examples:
{examples}

Test Input:
{json.dumps(test_input)}

What should be the Output grid? Only provide the output grid as your answer."""
                
                response = json.dumps(correct_output)
                
                # Format as conversation
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                
                # Tokenize conversation
                formatted_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                
                # Only process examples that fit within max_length after truncation
                tokenized = self.tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=self.max_length)
                
                # Skip overly long sequences
                if tokenized.input_ids.shape[1] >= self.max_length:
                    print(f"Skipping example with length {tokenized.input_ids.shape[1]} > {self.max_length}")
                    continue
                
                # Find the prompt length (where assistant response starts)
                user_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                user_tokenized = self.tokenizer(user_text, return_tensors="pt", truncation=True, max_length=self.max_length)
                prompt_length = user_tokenized.input_ids.shape[1]
                
                samples.append({
                    'input_ids': tokenized.input_ids.squeeze(),
                    'prompt_length': prompt_length
                })
        
        print(f"Prepared {len(samples)} training examples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # Pad sequences to the same length
    max_length = max(item['input_ids'].shape[0] for item in batch)
    
    input_ids = []
    prompt_lengths = []
    
    for item in batch:
        # Pad with EOS token
        padded = torch.full((max_length,), EOS_TOKEN_ID, dtype=torch.long)
        padded[:item['input_ids'].shape[0]] = item['input_ids']
        input_ids.append(padded)
        prompt_lengths.append(item['prompt_length'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'prompt_lengths': torch.tensor(prompt_lengths)
    }

def forward_process(x, p_data):
    """Forward process to add noise to the data."""
    t = torch.FloatTensor(x.shape[0]).uniform_(0, 1).to(x.device)
    
    # Implement q_t|0 according to equation (7)
    # This is a simplified version - adjust based on your exact noise schedule
    mask_prob = t.unsqueeze(-1).expand_as(x)
    mask = torch.rand_like(x, dtype=torch.float32) < mask_prob
    
    noisy_x = x.clone()
    noisy_x[mask] = MASK_ID
    
    return noisy_x, mask_prob

def train_sft(model, tokenizer, train_data, solutions, device, 
              epochs=3, batch_size=1, learning_rate=1e-5, 
              gradient_accumulation_steps=16, checkpoint_dir="checkpoints",
              save_steps=100):
    """Implements Algorithm 2: Supervised Fine-Tuning of LLaDA with memory optimizations"""
    
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = ARCDataset(train_data, solutions, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer with correct hyperparameters for 8B model
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.95),  # Better beta values for large models
        eps=1e-8,           # Epsilon stability value
        weight_decay=0.01   # Standard weight decay
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Learning rate scheduler
    total_steps = len(dataloader) * epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            input_ids = batch['input_ids'].to(device)
            prompt_lengths = batch['prompt_lengths'].to(device)
            
            # Forward process (add noise)
            noisy_batch, p_mask = forward_process(input_ids, None)
            
            # Do not add noise to the prompt
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]
            
            # Calculate the answer length
            prompt_mask = prompt_mask.to(torch.int64)
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
            
            # Forward pass with mixed precision
            with autocast():
                # Calculate loss
                masked_indices = (noisy_batch == MASK_ID)
                logits = model(input_ids=noisy_batch).logits
                
                token_loss = F.cross_entropy(
                    logits[masked_indices],
                    input_ids[masked_indices],
                    reduction='none'
                ) / p_mask[masked_indices]
                
                ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
                
                # Scale the loss by gradient accumulation steps
                ce_loss = ce_loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(ce_loss).backward()
            
            # Only update weights after accumulating gradients for specified steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                # Unscale the gradients for the optimizer
                scaler.unscale_(optimizer)
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights and optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                
                global_step += 1
                
                # Print progress
                if (batch_idx + 1) % (50 * gradient_accumulation_steps) == 0:
                    print(f"Batch {batch_idx + 1}, Loss: {ce_loss.item() * gradient_accumulation_steps:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    print(f"Saving checkpoint at step {global_step}")
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
            
            total_loss += ce_loss.item() * gradient_accumulation_steps
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_checkpoint_path, exist_ok=True)
        model.save_pretrained(epoch_checkpoint_path)
        tokenizer.save_pretrained(epoch_checkpoint_path)
    
    return model

# Load the generate function from llada_base.py
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

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0.,
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

    # Use half precision for inference
    model.half()
    
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
                # Free up memory before inference
                torch.cuda.empty_cache()
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

# Main execution
if __name__ == "__main__":
    # Print command line arguments
    print("Training arguments:")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Gradient accumulation steps: {args.grad_accum}")
    print(f"Maximum sequence length: {args.max_length}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Save checkpoint steps: {args.save_steps}")
    print()
    
    # Use less memory during model loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load model with better memory settings
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory savings
        use_cache=False,
        low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
        device_map="auto",           # Let HF decide optimal device mapping
        offload_folder="offload"     # Offload weights if needed
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    device = model.device
    print("Loaded model on device:", device)

    # Load data from JSON files
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_challenges.json', 'r') as f:
        data = json.load(f)
    
    with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    print(f"Loaded {len(data)} training tasks")
    
    # Perform SFT fine-tuning with memory optimizations
    print("Starting SFT fine-tuning...")
    model = train_sft(
        model, tokenizer, data, solutions, device, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        save_steps=args.save_steps
    )
    
    # Save the final fine-tuned model
    print("Saving final model...")
    model.save_pretrained("llada_finetuned_final")
    tokenizer.save_pretrained("llada_finetuned_final")
    print("Fine-tuned model saved to 'llada_finetuned_final'")
    
    # Test with the first example (optional, can be removed to save memory)
    print("Running inference test on first example...")
    
    # Clear cache before inference
    torch.cuda.empty_cache()
    gc.collect()
    
    first_key = list(data.keys())[0]
    task_data = data[first_key]

    train = task_data['train']
    test = task_data['test']
    test_input = test[0]['input']

    # Build training examples (same as original)
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
    input_ids = tokenizer(formatted_prompt, truncation=True, max_length=384)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Generate with the fine-tuned model - reduce generation length for memory
    with torch.cuda.amp.autocast():  # Use mixed precision
        out = generate(model, input_ids, steps=64, gen_length=64, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

    # Decode the output
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("\nGenerated output:")
    print(generated_text)

    # Get the correct output for this task
    correct_output = solutions[first_key][0]  # First solution for this task

    # Function to extract grid from generated text (same as original)
    def extract_grid_from_text(text):
        """Extract a grid from the generated text."""
        try:
            # Try to find JSON-like structure in the text
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

    # Extract generated grid
    generated_grid = extract_grid_from_text(generated_text)

    # Function to compare grids (same as original)
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

    # Calculate accuracy
    accuracy = compare_grids(generated_grid, correct_output)

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Task ID: {first_key}")
    print(f"\nCorrect Output Shape: {len(correct_output)}x{len(correct_output[0]) if correct_output else 0}")
    if generated_grid:
        print(f"Generated Output Shape: {len(generated_grid)}x{len(generated_grid[0]) if generated_grid else 0}")
    else:
        print("Generated Output Shape: Could not extract grid")

    # Display functions (same as original)
    def display_grid_horizontally(grid, title):
        """Display a grid in a horizontal format."""
        if not grid:
            print(f"{title}: Empty grid")
            return

        print(f"\n{title}:")
        for row in grid:
            print(" ".join(map(str, row)))

    # Display both grids horizontally
    display_grid_horizontally(correct_output, "Correct Output")
    display_grid_horizontally(generated_grid, "Generated Grid")

    print(f"\nACCURACY: {accuracy:.2f}%")