#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --ntasks=1            
#SBATCH --time=00:30:00       
#SBATCH --partition=eqa-cs6414   
#SBATCH --account=parallelcomputing

./original input.txt 0.5
./operation input.txt 0.5
./loop input.txt 0.5
./branch input.txt 0.5
./final input.txt 0.5
