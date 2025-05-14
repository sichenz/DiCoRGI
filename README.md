# DiCoRGI

## Environment Setup

First, you need to set up the conda environment on NYU GREENE:

1. Read the `JUPYTER.md` file for detailed instructions on setting up the conda environment on NYU GREENE
   ```bash
   cat JUPYTER.md
   ```

2. Follow the instructions in `JUPYTER.md` to create and activate your conda environment

## Package Installation

After setting up the conda environment, install the required packages:

1. Run the `req.sbatch` script to download and install all necessary packages for the conda environment
   ```bash
   sbatch req.sbatch
   ```

2. Check the status of your job using `squeue -u $USER` to ensure it completes successfully

## Data Generation

Generate the training data:

1. Navigate to the `data_gen` folder
   ```bash
   cd data_gen
   ```

2. Run the `rearc.sbatch` script to generate 10,000 examples for each of the 400 ARC problems
   ```bash
   sbatch rearc.sbatch
   ```

3. This process may take some time to complete. You can monitor the job status using:
   ```bash
   squeue -u $USER
   ```

## Data Verification

Verify that the data generation was successful:

1. Open and run the `visualization.ipynb` notebook to inspect the generated data
   ```bash
   jupyter notebook visualization.ipynb
   ```

2. Ensure that the notebook shows the correct number of examples (10K for each problem) and that the data format is as expected

## Model Training

After data generation and verification, proceed with the model training:

1. Navigate to the `llada` folder
   ```bash
   cd ../llada
   ```

2. Run the parallel processing script
   ```bash
   sbatch llada_parallel.sbatch
   ```

3. After the parallel processing completes, run the Supervised Fine-Tuning (SFT) script
   ```bash
   sbatch llada_sft.sbatch
   ```

4. Monitor both jobs using `squeue -u $USER` and check the output logs for any errors

## Troubleshooting

If you encounter any issues:

1. Check the job output logs in the slurm output files (typically named as `slurm-JOBID.out`)
2. Verify that all paths in the sbatch scripts are correct
3. Ensure that the conda environment is properly activated in each sbatch script
4. Check for sufficient disk space and compute resources

## Notes

- The data generation step (`rearc.sbatch`) creates 10,000 examples for each of the 400 ARC problems, which will require significant disk space
- The `llada_parallel.sbatch` and `llada_sft.sbatch` scripts will utilize GPU resources, so make sure your allocation has sufficient GPU time available
- Depending on your resource allocation, you may need to adjust the resource requests in the sbatch scripts