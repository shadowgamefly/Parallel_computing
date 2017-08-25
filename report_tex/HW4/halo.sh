#!/bin/bash
#SBATCH --nodelist=hermes1
#SBATCH --exclusive

# Wrote by Yijie Sun (ys7va) 2017.05.04
# the script is used to run halo simulation on Hermes
# the four inputs the program takes is num_thread, dim_x, dim_y and iterations
# details can be found in halo.

gcc -O3 -o halo halo.c -lpthread -lnuma
./halo 64 10000 10000 10000
