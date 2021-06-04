#!/bin/bash
#SBATCH -J countsketch_fbk_101
#SBATCH -o countsketch_fbk_101.out
#SBATCH -e countsketch_fbk_101.err
#SBATCH -p shared
#SBATCH -n 101
#SBATCH -c 1
#SBATCH -t 1600
#SBATCH --mem-per-cpu=32000
srun -n $SLURM_NTASKS --mpi=pmi2 python countsketch_resnet.py
