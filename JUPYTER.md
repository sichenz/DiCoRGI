
# NYU Greene Conda Environment Setup with Jupyter Notebook Support

This guide provides step-by-step instructions to set up a Conda + Jupyter notebook environment inside a Singularity container. 

## Setup Instructions

### 1. Create Environment Directory and Prepare Overlay

Create a dedicated directory for your environment:
```bash
mkdir /scratch/$USER/jupyter_env
cd /scratch/$USER/jupyter_env
```

See available overlay file systems:

```bash
ls /scratch/work/public/overlay-fs-ext3
```

Copy the overlay file system you need:

```bash
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
gunzip overlay-50G-10M.ext3.gz
```

### 2. Launch Singularity Container (Read/Write Overlay)

List available Singularity images: 

```bash
ls /scratch/work/public/singularity/
```

Start the Singularity container using the overlay filesystem in read/write mode:

```bash
singularity exec --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash
```

### 3. Install Miniforge (Conda)

Within the container, download and install Miniforge to set up a Conda environment:

```bash
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
sh Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
```

### 4. Create and Configure the Environment Script

Create an environment setup script to configure Conda:

1. Create and open the file for editing:

   ```bash
   touch /ext3/env.sh
   nano /ext3/env.sh
   ```

2. Add the following content:

   ```bash
   #!/bin/bash
   unset -f which
   source /ext3/miniforge3/etc/profile.d/conda.sh
   export PATH=/ext3/miniforge3/bin:$PATH
   export PYTHONPATH=/ext3/miniforge3/bin:$PATH
   ```

3. Save and close the file.

### 5. Initialize and Update Conda

Run the environment script and update Conda, then install necessary packages:

```bash
source /ext3/env.sh
conda config --remove channels defaults #usually it will say CondaKeyError: 'channels': value 'defaults' not present in config, this is normal! No further action needed 
conda update -n base conda -y
conda clean --all --yes
conda install pip --yes
conda install ipykernel --yes
exit
```

### 6. Creating and Activating a New Conda Environment

```bash
cd /scratch/$USER/jupyter_env

# Use previous Singularity session
singularity exec --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash

# Load environment variables
source /ext3/env.sh

# Create a new environment
conda create --name arc python=3.10
conda activate arc
pip install -r /scratch/$USER$/DiCoRGI/requirements.txt
exit
```

### 7. Download Packages into the Conda Environment

If the `pip install` command above does not work due to login node compute constraints, you can create an sbatch job similar to `req.sbatch` to run the package installations:

```bash
sbatch req.sbatch
squeue -u $USER
```

### 8. Configure the Jupyter Kernel for OnDemand Access

#### a. Set Up Kernel Directory

Create a directory for your custom Jupyter kernel and copy the kernel template:

```bash
mkdir -p ~/.local/share/jupyter/kernels
cd ~/.local/share/jupyter/kernels
cp -R /share/apps/mypy/src/kernel_template ./jupyter_env  # "jupyter_env" should be your Singularity env name
cd ./jupyter_env
ls
```

#### b. Edit the Kernel Launch Script

Open the `python` script for editing:

```bash
nano python
```

Make the following changes to the last few lines in the `python` script 

```bash
singularity exec $nv \
  --overlay /scratch/$USER/jupyter_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "source /ext3/env.sh; $cmd $args"
```

#### c. Configure `kernel.json`

Open `kernel.json` for editing:

```bash
nano kernel.json
```

Replace its contents with the following JSON, ensuring you update `<Your NetID>` with your actual NetID:

```json
{
  "argv": [
    "/home/<Your NetID>/.local/share/jupyter/kernels/jupyter_env/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "jupyter_env",
  "language": "python"
}
```


Happy computing!
