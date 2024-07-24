#!/bin/bash
#SBATCH --job-name=plot_submit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=13GB
#SBATCH --output=plot_submit.log

ml mamba && conda activate rombus

mpirun python make_plots.py -as True -i bbh_to_bns.ini