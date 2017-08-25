#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --exclusive

# cude.sh
# written by Jerry Sun (ys7va) 2017.05.09
# For submitting jobs through slurm
nvcc -O3 -o matrix matrix_mult.cu
./matrix 10000
