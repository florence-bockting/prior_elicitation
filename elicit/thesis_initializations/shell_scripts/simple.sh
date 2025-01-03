#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-15
#SBATCH --job-name=simple
#SBATCH --output=/work/mflobock/elicit/thesis_initializations/simple.out.txt
#SBATCH --mem=60000

module load nvidia/cuda
module load python/3.11.7-gcc114-base

for METHOD in sobol lhs random
do
    for LOSS in 0 10 30
    do
        for ITER in 1 32 128
        do
           srun python elicit/thesis_initializations/simulation_scripts/simple.py $SLURM_ARRAY_TASK_ID $METHOD $LOSS $ITER
        done
    done
done