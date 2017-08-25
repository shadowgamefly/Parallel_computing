#!/bin/bash
#SBATCH --nodes=20
#SBATCH --ntasks=200
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=parallelcomputing
#SBATCH --exclusive

module load openmpi

mpicc -O3 -o halo halo.c
mpirun -np 200 halo 10000 10000 1 10000 10000 1

