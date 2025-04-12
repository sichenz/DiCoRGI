#!/usr/bin/env python3
import os
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import argparse
import logging
from pathlib import Path
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARCDataset(Dataset):
    """Dataset for ARC tasks in the LLaDA format."""
    
    def __init__(self, data_dir, max_seq_len=512, max_examples_per_file=None):
        """
        Args:
            data_dir: Directory containing ARC task JSON files
            max_seq_len: Maximum sequence length for padding
            max_examples_per_file: Optional limit on number of examples per file (for testing)
        """
        self.data = []
        self.max_seq_len = max_seq_len
        
        # Load all JSON files in the directory
        file_count = 0
        example_count = 0
        
        for file_path in Path(data_dir).glob('*.json'):
            file_count += 1
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                
                # Apply limit if specified
                if max_examples_per_file is not None:
                    file_data = file_data[:max_examples_per_file]
                
                self.data.extend(file_data)
                example_count += len(file_data)
                
                # Log progress occasionally
                if file_count % 50 == 0:
                    logger.info(f"Loaded {example_count} examples from {file_count} files so far...")
        
        logger.info(f"Loaded {len(self.data)} examples from {file_count} files in {data_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        input_grid = example['input']
        output_grid = example['output']
        
        # Flatten the grids for processing
        input_flat = [item for row in input_grid for item in row]
        output_flat = [item for row in output_grid for item in row]
        
        # Prepare sequences for LLaDA
        # We'll combine input and output with a special token in between
        combined = input_flat + [10] + output_flat  # Using 10 as a separator token
        
        # Get dimensions for reconstruction later
        input_shape = (len(input_grid), len(input_grid[0]))
        output_shape = (len(output_grid), len(output_grid[0]))
        
        return {
            'combined': torch.tensor(combined, dtype=torch.long),
            'input_length': len(input_flat),
            'output_length': len(output_flat),
            'input_shape': input_shape,
            'output_shape': output_shape
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    max_len = max([len(item['combined']) for item in batch])
    
    # Pad sequences
    combined_padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = len(item['combined'])
        combined_padded[i, :seq_len] = item['combined']
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (combined_padded != 0).float()
    
    return {
        'combined': combined_padded,
        'attention_mask': attention_mask,
        'input_lengths': torch.tensor([item['input_length'] for item in batch]),
        'output_lengths': torch.tensor([item['output_length'] for item in batch]),
        'input_shapes': [item['input_shape'] for item in batch],
        'output_shapes': [item['output_shape'] for item in batch]
    }

class TransformerEncoder(nn.Module):
    """Transformer encoder for LLaDA mask predictor."""
    
    def __init__(self, vocab_size=11, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor [batch_size, seq_len]
            src_mask: Mask for the encoder
            src_key_padding_mask: Padding mask for the encoder
        """
        # Embedding
        src = self.embedding(src) * (self.d_model ** 0.5)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LLaDA(nn.Module):
    """Large Language Diffusion with Masking model."""
    
    def __init__(self, vocab_size=11, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(LLaDA, self).__init__()
        
        # The mask token is the last token in the vocabulary
        self.mask_token_id = vocab_size - 1
        self.vocab_size = vocab_size
        
        # Mask predictor (transformer encoder)
        self.mask_predictor = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, x, mask_ratio, attention_mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len]
            mask_ratio: Ratio of tokens to mask (scalar between 0 and 1)
            attention_mask: Attention mask (1 for real tokens, 0 for padding)
        """
        batch_size, seq_len = x.size()
        device = x.device
        
        # Create a copy of input for prediction targets
        targets = x.clone()
        
        # Create a mask to indicate which tokens to mask (1 means mask, 0 means keep)
        # Only mask real tokens (not padding)
        if attention_mask is not None:
            # Create a mask for tokens that could be masked (not padding)
            maskable = attention_mask.bool()
            
            # Initialize mask with zeros
            mask = torch.zeros_like(x, dtype=torch.bool)
            
            # For each sequence in the batch
            for i in range(batch_size):
                # Count real tokens in this sequence
                real_tokens = maskable[i].sum().item()
                
                # Calculate how many tokens to mask for this sequence
                num_masks = int(real_tokens * mask_ratio)
                
                # Get indices of real tokens
                real_token_indices = torch.where(maskable[i])[0]
                
                # Randomly select tokens to mask
                if num_masks > 0 and len(real_token_indices) > 0:
                    mask_indices = real_token_indices[torch.randperm(len(real_token_indices))[:num_masks]]
                    mask[i, mask_indices] = True
        else:
            # Randomly sample mask indices for all tokens
            mask = torch.bernoulli(torch.full((batch_size, seq_len), mask_ratio)).bool().to(device)
        
        # Apply masking to input
        masked_input = x.clone()
        masked_input[mask] = self.mask_token_id
        
        # Get predictions from the model
        logits = self.mask_predictor(masked_input)
        
        # We only compute loss on masked tokens
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Scale loss by mask ratio (as in the paper)
        mask_sum = mask.float().sum() + 1e-10  # Add small constant to avoid division by zero
        loss = (loss * mask.float()).sum() / mask_sum / mask_ratio
        
        return {
            'loss': loss,
            'logits': logits,
            'mask': mask
        }
    
    def sample(self, prompt=None, output_length=100, steps=50, temperature=1.0, device='cuda'):
        """
        Sample from the model using the diffusion process.
        
        Args:
            prompt: Input tensor [batch_size, prompt_len] or None
            output_length: Length of output to generate
            steps: Number of diffusion steps
            temperature: Temperature for sampling
            device: Device for computation
        """
        batch_size = 1 if prompt is None else prompt.size(0)
        
        # Initialize with all masked tokens for the output part
        if prompt is None:
            # If no prompt, generate a sequence of all mask tokens
            x = torch.full((batch_size, output_length), self.mask_token_id, device=device)
        else:
            # Concatenate prompt with masked output space
            prompt_len = prompt.size(1)
            output_mask = torch.full((batch_size, output_length), self.mask_token_id, device=device)
            x = torch.cat([prompt, output_mask], dim=1)
            
        # Simulate the reverse process (from t=1 to t=0)
        for step in range(steps, 0, -1):
            # Current mask ratio
            t = step / steps
            s = (step - 1) / steps  # Next step's mask ratio
            
            # Predict all masked tokens
            with torch.no_grad():
                logits = self.mask_predictor(x)
                
            # Apply temperature for sampling
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Sample from the predicted distributions
            pred_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, -1)
            
            # Update currently masked tokens
            mask = (x == self.mask_token_id)
            x = x.clone()
            x[mask] = pred_tokens[mask]
            
            # For the next step, we need to remask some tokens to follow the diffusion process
            # Skip remasking on the last step
            if step > 1:
                # Calculate how many tokens to remask
                remask_ratio = s / t
                
                # Only consider output tokens for remasking if there's a prompt
                if prompt is not None:
                    remask_range = slice(prompt_len, x.size(1))
                else:
                    remask_range = slice(0, x.size(1))
                
                # Randomly select tokens to remask
                remask = torch.bernoulli(
                    torch.full((batch_size, x.size(1) - prompt_len if prompt is not None else x.size(1)), remask_ratio)
                ).bool().to(device)
                
                # Expand to full sequence size
                if prompt is not None:
                    full_remask = torch.zeros_like(x, dtype=torch.bool)
                    full_remask[:, prompt_len:] = remask
                    remask = full_remask
                
                # Apply remasking
                x[remask] = self.mask_token_id
        
        # Return the generated output
        if prompt is not None:
            return x[:, prompt_len:]
        else:
            return x

class ARCDatasetSubset(Dataset):
    """Memory efficient subset of the ARC dataset for large scale training"""
    
    def __init__(self, data_dir, files_per_subset=10, max_examples_per_file=None):
        """
        Args:
            data_dir: Directory containing ARC task JSON files
            files_per_subset: Number of files to load at once
            max_examples_per_file: Optional limit on examples per file
        """
        self.data_dir = data_dir
        self.files_per_subset = files_per_subset
        self.max_examples_per_file = max_examples_per_file
        
        # Get all JSON files in the directory
        self.file_paths = list(Path(data_dir).glob('*.json'))
        self.file_count = len(self.file_paths)
        
        # Estimate total examples if we processed all files
        if max_examples_per_file is None:
            # Sample a few files to estimate average examples per file
            sample_size = min(10, self.file_count)
            example_counts = []
            for i in range(sample_size):
                with open(self.file_paths[i], 'r') as f:
                    example_counts.append(len(json.load(f)))
            avg_examples = sum(example_counts) / len(example_counts)
            self.estimated_total = int(avg_examples * self.file_count)
        else:
            self.estimated_total = self.file_count * max_examples_per_file
        
        # Start with the first subset
        self.current_subset_idx = 0
        self.load_current_subset()
    
    def load_current_subset(self):
        """Load the current subset of files into memory"""
        self.data = []
        
        # Determine start and end indices for current subset
        start_idx = self.current_subset_idx * self.files_per_subset
        end_idx = min(start_idx + self.files_per_subset, self.file_count)
        
        # Load files in the current subset
        for i in range(start_idx, end_idx):
            with open(self.file_paths[i], 'r') as f:
                file_data = json.load(f)
                
                # Apply limit if specified
                if self.max_examples_per_file is not None:
                    file_data = file_data[:self.max_examples_per_file]
                
                self.data.extend(file_data)
        
        logger.info(f"Loaded subset {self.current_subset_idx + 1}/{(self.file_count + self.files_per_subset - 1) // self.files_per_subset} with {len(self.data)} examples")
    
    def next_subset(self):
        """Move to the next subset of files"""
        self.current_subset_idx = (self.current_subset_idx + 1) % ((self.file_count + self.files_per_subset - 1) // self.files_per_subset)
        self.load_current_subset()
    
    def __len__(self):
        return self.estimated_total
    
    def __getitem__(self, idx):
        # Map the global index to a local index in the current subset
        local_idx = idx % len(self.data)
        
        # If we've gone through the entire subset, load the next one
        if local_idx == 0 and idx > 0:
            self.next_subset()
        
        example = self.data[local_idx]
        input_grid = example['input']
        output_grid = example['output']
        
        # Flatten the grids for processing
        input_flat = [item for row in input_grid for item in row]
        output_flat = [item for row in output_grid for item in row]
        
        # Prepare sequences for LLaDA
        combined = input_flat + [10] + output_flat  # Using 10 as a separator token
        
        # Get dimensions for reconstruction later
        input_shape = (len(input_grid), len(input_grid[0]))
        output_shape = (len(output_grid), len(output_grid[0]))
        
        return {
            'combined': torch.tensor(combined, dtype=torch.long),
            'input_length': len(input_flat),
            'output_length': len(output_flat),
            'input_shape': input_shape,
            'output_shape': output_shape
        }

def train_llada(args):
    """Train the LLaDA model on ARC tasks."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and device.type == 'cuda':
        logger.info("Using memory-efficient attention")
    
    # Prepare datasets - use memory-efficient subset loader for large datasets
    logger.info("Setting up dataset...")
    if args.memory_efficient:
        train_dataset = ARCDatasetSubset(
            args.train_data_dir,
            files_per_subset=args.files_per_subset,
            max_examples_per_file=args.max_examples_per_file
        )
    else:
        train_dataset = ARCDataset(
            args.train_data_dir,
            max_examples_per_file=args.max_examples_per_file
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Calculate the maximum vocabulary size needed
    # ARC tasks use integers from 0-9 for grid values, plus we need a mask token (10)
    vocab_size = 11
    
    # Initialize model
    logger.info(f"Initializing model with d_model={args.d_model}, layers={args.num_layers}")
    model = LLaDA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # Use mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing")
        model.mask_predictor.transformer_encoder.layers.apply(
            lambda module: setattr(module, "_checkpoint", True) if hasattr(module, "_checkpoint") else None
        )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize learning rate scheduler
    total_steps = len(train_loader) * args.num_epochs if not args.memory_efficient else args.num_epochs * (train_dataset.estimated_total // args.batch_size)
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    # Load checkpoint if provided
    if args.resume_from_checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint.get('global_step', 0)
        logger.info(f"Resumed from step {global_step}")
    
    try:
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                combined = batch['combined'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Sample a random mask ratio for this batch
                mask_ratio = random.uniform(0.0, 1.0)
                
                # Forward pass with mixed precision if enabled
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(combined, mask_ratio, attention_mask)
                        loss = outputs['loss']
                else:
                    outputs = model(combined, mask_ratio, attention_mask)
                    loss = outputs['loss']
                
                # Backward pass with mixed precision if enabled
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # Update progress
                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save checkpoint occasionally
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # Save model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item()
                    }, os.path.join(checkpoint_path, "model.pt"))
                    
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Clean GPU memory periodically
                if args.clean_memory_steps > 0 and global_step % args.clean_memory_steps == 0:
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_epoch_loss
                }, os.path.join(args.output_dir, "best_model.pt"))
                logger.info(f"Saved best model with loss: {best_loss:.4f}")
            
            # Save epoch checkpoint
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss
            }, os.path.join(args.output_dir, f"epoch_{epoch+1}_model.pt"))
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model...")
        torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(args.output_dir, "interrupted_model.pt"))
        logger.info("Model saved.")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    logger.info("Training completed!")
    
    return model

def evaluate_llada(model, test_data_dir, device='cuda', steps=50, temperature=1.0, batch_size=32, max_examples_per_file=None):
    """Evaluate LLaDA model on ARC test tasks."""
    # Load test dataset
    test_results = {}
    total_correct = 0
    total_examples = 0
    
    # Set model to evaluation mode
    model.eval()
    
    # Process each file in the test directory
    for file_path in sorted(Path(test_data_dir).glob('*.json')):
        task_id = file_path.stem
        
        with open(file_path, 'r') as f:
            task_data = json.load(f)
        
        # Apply limit if specified
        if max_examples_per_file is not None:
            task_data = task_data[:max_examples_per_file]
        
        task_correct = 0
        task_total = len(task_data)
        
        # Process examples in batches
        for start_idx in tqdm(range(0, task_total, batch_size), desc=f"Evaluating {task_id}"):
            end_idx = min(start_idx + batch_size, task_total)
            batch_examples = task_data[start_idx:end_idx]
            
            for example in batch_examples:
                input_grid = example['input']
                correct_output_grid = example['output']
                
                # Flatten the input grid for processing
                input_flat = torch.tensor([item for row in input_grid for item in row], dtype=torch.long).unsqueeze(0).to(device)
                
                # Determine output grid dimensions
                output_height = len(correct_output_grid)
                output_width = len(correct_output_grid[0]) if output_height > 0 else 0
                output_size = output_height * output_width
                
                # Add separator token
                input_with_sep = torch.cat([input_flat, torch.tensor([[10]], device=device)], dim=1)
                
                # Generate output
                with torch.no_grad():
                    generated_flat = model.sample(
                        prompt=input_with_sep,
                        output_length=output_size,
                        steps=steps,
                        temperature=temperature,
                        device=device
                    )
                
                # Reshape the generated output to match the expected grid
                generated_grid = generated_flat.cpu().numpy()[0].reshape(output_height, output_width).tolist()
                
                # Compare with the correct output
                if generated_grid == correct_output_grid:
                    task_correct += 1
                    total_correct += 1
                
                total_examples += 1
            
            # Clean GPU memory after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate accuracy for this task
        task_accuracy = task_correct / task_total if task_total > 0 else 0
        test_results[task_id] = {
            'correct': task_correct,
            'total': task_total,
            'accuracy': task_accuracy
        }
        
        logger.info(f"Task {task_id} - Accuracy: {task_accuracy:.4f} ({task_correct}/{task_total})")
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_examples})")
    
    # Save results to a JSON file
    result_path = os.path.join(os.path.dirname(test_data_dir), "evaluation_results.json")
    with open(result_path, 'w') as f:
        json.dump({
            'task_results': test_results,
            'overall': {
                'correct': total_correct,
                'total': total_examples,
                'accuracy': overall_accuracy
            }
        }, f, indent=2)
    
    logger.info(f"Saved results to {result_path}")
    
    return test_results, overall_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LLaDA model on ARC tasks")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, default="/scratch/sz4972/DiCoRGI/llada/llada_data/training",
                        help="Directory containing training data")
    parser.add_argument("--test_data_dir", type=str, default="/scratch/sz4972/DiCoRGI/llada/llada_data/testing",
                        help="Directory containing test data")
    parser.add_argument("--output_dir", type=str, default="./llada_arc_model",
                        help="Directory to save model checkpoints")
    parser.add_argument("--max_examples_per_file", type=int, default=None,
                        help="Maximum number of examples to load per file (for testing)")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of model embeddings")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of total training steps for warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for data loading")
    
    # Memory efficiency arguments
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="Use memory-efficient dataset loading")
    parser.add_argument("--files_per_subset", type=int, default=10,
                        help="Number of files to load at once in memory-efficient mode")
    parser.add_argument("--clean_memory_steps", type=int, default=500,
                        help="Clean GPU memory every X steps (0 to disable)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    
    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Number of diffusion steps for evaluation")
    parser.add_argument("--eval_temperature", type=float, default=0.8,
                        help="Temperature for evaluation sampling")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation on a saved model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model for evaluation")
    
    args = parser.parse_args()
    
    if args.eval_only and args.model_path is not None:
        # Load the saved model and evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the model with the same architecture
        model = LLaDA(
            vocab_size=11,  # ARC tasks use integers 0-9 plus mask token
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        ).to(device)
        
        # Load the saved state dict
        checkpoint = torch.load(args.model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Evaluate the model
        evaluate_llada(
            model=model,
            test_data_dir=args.test_data_dir,
            device=device,
            steps=args.eval_steps,
            temperature=args.eval_temperature,
            batch_size=args.batch_size,
            max_examples_per_file=args.max_examples_per_file
        )
    else:
        # Train the model and then evaluate
        model = train_llada(args)
        
        # Evaluate on test data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate_llada(
            model=model,
            test_data_dir=args.test_data_dir,
            device=device,
            steps=args.eval_steps,
            temperature=args.eval_temperature,
            batch_size=args.batch_size,
            max_examples_per_file=args.max_examples_per_file
        )