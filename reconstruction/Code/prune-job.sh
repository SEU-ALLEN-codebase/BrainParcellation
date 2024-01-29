#!/bin/bash
#SBATCH --job-name=Ultimate230k
#SBATCH --exclude=admin
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=20
#SBATCH --partition=normal
#SBATCH --chdir=/public/home/vkzohj/ultimate_230k


vaa3d=../software/vaa3d/start_vaa3d.sh
python=/public/home/vkzohj/.conda/envs/230k/bin/python

for i in `seq 0 29`; do
    srun --ntasks=1 --cpus-per-task=20 --nodes=1 \
        python seg_prune.py `printf "batch/%02d" ${i}` pruned app2 &
done