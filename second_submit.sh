#!/bin/bash
#SBATCH --job-name=second_submit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=64GB
#SBATCH --output=second_submit.log

ml mamba && conda activate rombus

python build_second.py -as True -i bbh_to_bns.ini
#python build_second.py -as True -i bns_to_bbh.ini