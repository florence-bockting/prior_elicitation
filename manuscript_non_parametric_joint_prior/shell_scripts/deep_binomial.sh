#!/bin/bash -l
#SBATCH --partition=gpu_med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --job-name=independent_binomial
#SBATCH --output=/work/mflobock/elicit/simulations/deep_prior_examples/independent_binomial.out.txt
#SBATCH --mem=60000

module load nvidia/cuda
module load python/3.11.7-gcc114-base

srun python elicit/simulations/deep_prior_examples/independent_binomial.py $SLURM_ARRAY_TASK_ID
