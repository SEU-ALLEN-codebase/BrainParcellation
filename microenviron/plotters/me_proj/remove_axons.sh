
# Extract the dendrites from each neuron and remove files without dendrites
dataset=ion_hip_2um
for fpath in `ls /data/lyf/data/fullNeurons/all_neurons/${dataset}/*swc`; do fn=$(basename $fpath); awk '$2 != 2' $fpath > /data/lyf/data/fullNeurons/all_neurons_dendrites/${dataset}/${fn}; echo $fn; done
# remove empty file
find /data/lyf/data/fullNeurons/all_neurons_dendrites/${dataset} -type f -exec awk 'END {if (NR<3) exit 0; exit 1}' {} \; -exec rm {} \;


#for fpath in `ls /PBshare/SEU-ALLEN/Users/SD-Jiang/transfer/ION_Hipp_10100/all_raw/*swc`; do fn=$(basename $fpath); awk '$2 == 1 || $2 == 2' $fpath > ./swc_axons/${fn}; echo $fn; done
