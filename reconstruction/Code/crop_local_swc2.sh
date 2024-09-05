#!/bin/bash

indir=$1
outdir=$2

if [ ! -d ${outdir} ];then
    mkdir -p ${outdir}
fi

#multi-thread
threadNum=20
tempfifo="L0_volumes_fea"
mkfifo ${tempfifo}
exec 6<>${tempfifo}
rm -f ${tempfifo}
for ((i=1;i<${threadNum};i++))
do
{
    echo ;
}
done >&6  

# for folder in $(ls ${indir})
# do
    # folder_path=${indir}/${folder}
    # out_folder=${outdir}/${folder}
    # if [[ ! -d ${out_folder} ]]
    # then 
    #     mkdir $out_folder
    # fi

for swc in $(find ${indir}/ -name '*swc')
do
read -u6
{
    # echo ${vol}
    #swc_out=${outdir}/${folder}/${swc##*/}
    swc_out=${outdir}/${swc##*/}
    if [[ ! -f ${swc_out} ]];then
        vaa3d -x NeuroMorphoLib -f crop_local_swc -i ${swc} -o ${swc_out} -p 100
    fi
    echo >&6
} &
done       
# done
wait
exec 6>&-
