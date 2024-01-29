#!/bin/bash
#SBATCH --job-name=Ultimate230k
#SBATCH --exclude=admin
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --array=0-19
#SBATCH --partition=normal
#SBATCH --chdir=/public/home/vkzohj/ultimate_230k

python=/public/home/vkzohj/.conda/envs/230k/bin/python
cat `printf "batch/%02d" ${SLURM_ARRAY_TASK_ID}` | while read img;
do
    # check resource availability
    while [ "$(jobs -p | wc -l)" -ge "$SLURM_NTASKS" ]; do
        sleep 30
    done
    brain=`dirname ${img} | xargs basename`
    file=`basename -s .tif ${img}`
    outfile=filtered/${brain}/${file}.v3dpbd
    [ -f $outfile ] || srun --ntasks=1 --cpus-per-task=1 --nodes=1 $python filter.py $img $outfile &
done
wait