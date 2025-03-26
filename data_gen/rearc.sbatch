#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=rearc
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=rearc.out

module purge
cd /scratch/$USER/DiCoRGI/data_gen

# Download files only if they don't exist
files=(
    "https://raw.githubusercontent.com/michaelhodel/re-arc/main/dsl.py"
    "https://raw.githubusercontent.com/michaelhodel/re-arc/main/generators.py"
    "https://raw.githubusercontent.com/michaelhodel/re-arc/main/main.py"
    "https://raw.githubusercontent.com/michaelhodel/re-arc/main/utils.py"
    "https://raw.githubusercontent.com/michaelhodel/re-arc/main/verifiers.py"
)

for file_url in "${files[@]}"; do
    # Extract just the filename from the URL
    file_name=$(basename "$file_url")
    
    # Check if file exists
    if [ ! -f "$file_name" ]; then
        echo "Downloading $file_name..."
        wget "$file_url"
    else
        echo "$file_name already exists, skipping download."
    fi
done

# jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 gen10000.ipynb --output gen10000_executed.ipynb"

# Then run the notebook
singularity exec --nv \
  --overlay /scratch/$USER/jupyter_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate arc; cd /scratch/$USER/DiCoRGI/data_gen; python gen10000.py 2"   
