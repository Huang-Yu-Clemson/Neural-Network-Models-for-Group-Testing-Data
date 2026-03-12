#!/bin/bash
#SBATCH --job-name simulation
#SBATCH --time=24:00:00
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G


module load r/4.4.0
module load gcc/12.3.0
cd $SLURM_SUBMIT_DIR
data_model="M2"  # M1, M2, M3 or B, IM1, IM2
algorithm="L1"   # L1, L2, NN or CEL, WCEL

output_directory="${data_model}_${algorithm}/out/"
if [ ! -d "$output_directory" ]; then
  mkdir -p "$output_directory"
fi

for i in {1..500}
do
  srun --ntasks=1 --nodes=1 --mem-per-cpu=5G Rscript main.R $data_model $algorithm $i > "${output_directory}${data_model}_${algorithm}_$i.out" &
  sleep 1 # Add a 1-second delay between each srun command
done
wait