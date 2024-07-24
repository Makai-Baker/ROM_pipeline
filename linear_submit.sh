#!/bin/bash
#SBATCH --job-name=linear_submit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --output=linear_submit.log

ml mamba && conda activate rombus

#mpirun python build_linear.py -as True -n 30000 -i bns_to_bbh.ini
mpirun python build_linear.py -as True -n 50000 -i bbh_to_bns.ini
