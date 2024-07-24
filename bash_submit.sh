jid0=$(sbatch linear_submit.sh)
jid1=$(sbatch --dependency=afterok:${jid0##* } second_submit.sh)
sbatch --dependency=afterok:${jid1##* } plot_submit.sh
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"