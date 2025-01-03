#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-15
#SBATCH --job-name=simple_128
#SBATCH --output=/work/mflobock/elicit/thesis_initializations/simple_128.out.txt
#SBATCH --mem=60000

module load nvidia/cuda
module load python/3.11.7-gcc114-base

for METHOD in sobol lhs random
do
    for LOSS in 0 10 30
    do
        srun python elicit/thesis_initializations/simulation_scripts/simple_128.py $SLURM_ARRAY_TASK_ID $METHOD $LOSS
    done
done