#!/bin/bash
#SBATCH --job-name=Ultimate230k
#SBATCH --exclude=admin
#SBATCH --ntasks=30
#SBATCH --array=0-19
#SBATCH --partition=normal
#SBATCH --chdir=/public/home/vkzohj/ultimate_230k
#SBATCH --mem=0

vaa3d=../software/vaa3d/start_vaa3d.sh
python=/public/home/vkzohj/.conda/envs/230k/bin/python

cat `printf "batch/%02d" ${SLURM_ARRAY_TASK_ID}` | while read img;
do
    # check resource availability
    while [ "$(jobs -p | wc -l)" -ge "$SLURM_NTASKS" ]; do
        sleep 30
    done
    brain=`dirname ${img} | xargs basename`
    file=`basename -s .v3dpbd ${img}`
    outdir=app2/${brain}
    mkdir -p ${outdir}
    outfile=${outdir}/${file}.swc
    [ -f ${outfile} ] || srun --ntasks=1 --cpus-per-task=1 --nodes=1 --mem=0 -o /dev/null xvfb-run -a bash -c "
        tdir=\`mktemp -d\`
        trap 'rm -rf \$tdir' EXIT
        infile=\$tdir/${file}.v3dpbd
        cp ${img} \$infile
        
        ${python} soma_finder.py \$infile
        ${vaa3d} -x vn2 -f app2 -i \$infile -o \$infile.swc -p \$infile.marker 0 AUTO 0 1 1 0 5 1 0 0
        [ -f \$infile.swc ] && cp \$infile.swc ${outfile}
    " &
done
wait