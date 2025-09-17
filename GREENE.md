# NYU Greene HPC Setup Guide

This guide walks you through setting up and using NYU's Greene High Performance Computing cluster with Singularity containers for machine learning and data science projects.

## Prerequisites

- NYU HPC account with NetID
- Terminal access (MacBook Terminal, Windows Subsystem for Linux, etc.)
- Basic familiarity with Linux commands

## 1. Initial Connection

### SSH Access
```bash
# Connect to the gateway
ssh sz4972@gw.hpc.nyu.edu

# Connect to Greene cluster
ssh sz4972@greene.hpc.nyu.edu
```

### Troubleshooting Connection Issues
If you encounter "access blocked" errors:
```bash
ssh-keygen -R gw.hpc.nyu.edu
```

## 2. Initial Setup on Login Node

### Create Conda Environment (Optional - for login node only)
```bash
conda create -n nlp python=3.8
```

## 3. Singularity Container Setup

### Step 1: Explore Available Resources
```bash
# Check available overlay filesystems
ls /scratch/work/public/overlay-fs-ext3

# Check available Singularity images
ls /scratch/work/public/singularity/
```

### Step 2: Set Up Overlay Filesystem
```bash
# Navigate to your scratch directory
cd /scratch/$USER

# Copy and extract overlay filesystem
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
gunzip overlay-50G-10M.ext3.gz
```

### Step 3: Enter Singularity Container
```bash
singularity exec --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash
```

### Step 4: Install Miniforge (Conda Alternative)
```bash
# Download Miniforge
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Install Miniforge
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3

# Clean up
rm Miniforge3-Linux-x86_64.sh
```

### Step 5: Create Environment Setup Script
```bash
# Create environment script
touch /ext3/env.sh
nano /ext3/env.sh
```

Add the following content to `/ext3/env.sh`:
```bash
#!/bin/bash

unset -f which

source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
```

### Step 6: Configure Conda
```bash
# Source the environment
source /ext3/env.sh

# Remove default channels (normal to see error message)
conda config --remove channels defaults

# Update conda
conda update -n base conda -y

# Clean conda cache
conda clean --all --yes

# Install pip
conda install pip -y
```

## 4. Creating New Environments

### Navigate to Your Scratch Directory
```bash
cd /scratch/$USER
```

### Enter Container with Updated Singularity Image
```bash
singularity exec --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash
```

### Create and Activate Environment
```bash
# Source environment
source /ext3/env.sh

# Create new environment
conda create -n <environment_name> python=<python_version>

# Activate environment
conda activate <environment_name>
```

## 5. Installing Packages

### Using Alternative Container
```bash
cd /scratch/$USER

singularity exec --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash

source /ext3/env.sh
conda activate arc
cd /home/$USER/<project_directory>

# Install from requirements file
pip install -r requirements.txt

exit
```

## 6. Job Submission

### Create SBATCH Script
Create a file named `<job_name>.sbatch`:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=<job_name>
#SBATCH --mail-user=<your_email>
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=<job_name>.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER

singularity exec --nv \
    --overlay /scratch/$USER/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate <environment_name>; cd /home/$USER/<project_directory>; python3 main.py"
```

### Submit and Monitor Jobs
```bash
# Submit job
sbatch <job_name>.sbatch

# Check job status
squeue -u $USER
```

## 7. Interactive Sessions

### Request Interactive GPU Session
```bash
srun --gres=gpu:1 --cpus-per-task=1 --mem=16GB --time=00:30:00 --pty /bin/bash
```

### Enter Container in Interactive Mode
```bash
singularity exec --nv \
    --overlay jupyter_env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash
```

## Key Points to Remember

1. **Login Node vs Compute Nodes**: Only use login nodes for light tasks like file management and job submission
2. **Scratch Directory**: Always work in `/scratch/$USER` for better performance
3. **Overlay Filesystems**: These provide persistent storage within containers
4. **GPU Access**: Use `--nv` flag with Singularity for GPU access
5. **Environment Persistence**: Your conda environments and packages persist in the overlay filesystem
6. **Resource Limits**: Be mindful of time limits and resource allocation in SBATCH scripts

## Troubleshooting

- If conda commands fail, ensure you've sourced `/ext3/env.sh`
- For permission errors, check that you're in the correct directory (`/scratch/$USER`)
- For GPU issues, verify you're using the `--nv` flag with Singularity
- If jobs fail to start, check resource availability with `squeue` and `sinfo`

## Useful Commands Reference

```bash
# Check available resources
sinfo

# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check job details
scontrol show job <job_id>

# Monitor GPU usage (within container)
nvidia-smi
```
