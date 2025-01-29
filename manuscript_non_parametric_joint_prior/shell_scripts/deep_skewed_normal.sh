#!/bin/bash -l
#SBATCH --partition=gpu_med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-29
#SBATCH --job-name=deep_skewed_normal
#SBATCH --output=/work/mflobock/elicit/simulations/deep_prior_examples/skewed_normal.out.txt
#SBATCH --mem=60000

module load python/3.11.7-gcc114-base

srun python elicit/simulations/deep_prior_examples/scenarios_normal.py $SLURM_ARRAY_TASK_ID "skewed"
