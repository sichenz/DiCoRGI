import torch
import numpy as np
import torch.nn.functional as F
import json

from transformers import AutoModel, AutoTokenizer

# generate.py
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

# Load model and tokenizer
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, use_cache=False, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
print("Loaded model on device:", model.device)

# Load data from JSON file
with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_challenges.json', 'r') as f:
    data = json.load(f)

# Get the first key from the data (assuming you want to process the first example)
first_key = list(data.keys())[0]
task_data = data[first_key]

train = task_data['train']
test = task_data['test']
test_input = test[0]['input']

# Build training examples
examples = ""
for item in train:
    # Use correct dictionary key references
    input_grid = json.dumps(item['input'])
    output_grid = json.dumps(item['output'])
    examples += f"Input: {input_grid}\nOutput: {output_grid}\n\n"

print("Training examples:")
print(examples)

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
input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)

# Make sure you have the generate function imported or defined
# If not, you'll need to either import it or use model.generate() instead
out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

# Decode the output
generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
print("\nGenerated output:")
print(generated_text)

# Load correct solutions
with open('/scratch/sz4972/DiCoRGI/llada/arc_data/arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

# Get the correct output for this task
correct_output = solutions[first_key][0]  # First solution for this task

# Function to extract grid from generated text
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

# Function to compare grids
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

# Function to display grid horizontally
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

# Function to visualize grid side by side
def visualize_grids_side_by_side(grid1, grid2, title1="Correct Output", title2="Generated Grid"):
    """Display two grids side by side with color coding."""
    if not grid1 or not grid2:
        print("One or both grids are empty")
        return

    max_height = max(len(grid1), len(grid2))

    # Add padding to shorter grid
    padded_grid1 = grid1 + [[0] * len(grid1[0])] * (max_height - len(grid1)) if len(grid1) < max_height else grid1
    padded_grid2 = grid2 + [[0] * len(grid2[0])] * (max_height - len(grid2)) if len(grid2) < max_height else grid2

    print(f"\n{title1:<30} | {title2}")
    print("-" * 30 + " | " + "-" * 30)

    for i in range(max_height):
        if i < len(grid1):
            row1_str = " ".join(f"{cell:2d}" for cell in grid1[i])
        else:
            row1_str = "  " * len(grid2[0])

        if i < len(grid2):
            row2_str = " ".join(f"{cell:2d}" for cell in grid2[i])
        else:
            row2_str = "  " * len(grid2[0])

        print(f"{row1_str:<30} | {row2_str}")

# Display side by side comparison
visualize_grids_side_by_side(correct_output, generated_grid)