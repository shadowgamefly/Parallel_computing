#!/bin/bash
#SBATCH --nodes=20
#SBATCH --ntasks=200
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=parallelcomputing
#SBATCH --exclusive
#slurm script to run halo
#Jerry Sun (ys7va) 2017/4/12 
module load openmpi
#compile with -O3 flag output executable as halo
mpicc -O3 -o halo halo.c
#run halo in 200 tasks
#condition: 10000 iterations on 10000 * 10000 plate, 1 iter per cell, output 1 final image using 2 ghost-cell 
mpirun -np 200 halo 10000 10000 1 10000 10000 2

