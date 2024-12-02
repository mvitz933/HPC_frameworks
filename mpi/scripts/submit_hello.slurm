#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4000
#SBATCH --time=1-12:00:00
#SBATCH --partition=serial
#SBATCH --account=hpc-bootcamp

module load gcc openmpi
#module load anaconda2
#module load anaconda3


mpiexec ./mpihello-c
#mpiexec python mpi2.py

exit
