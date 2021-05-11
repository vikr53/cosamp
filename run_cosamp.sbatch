#!/bin/bash
#SBATCH -J cosamp_fbk
#SBATCH -o cosamp_fbk.out
#SBATCH -e cosamp_fbk.err
#SBATCH -p shared
#SBATCH -n 11
#SBATCH -t 1600
#SBATCH --mem-per-cpu=4000
srun -n $SLURM_NTASKS --mpi=pmi2 python cosamp_resnet.py
