input=/data/lyf/data/fullNeurons/all_neurons/ion_hip_2um
output=/data/lyf/data/fullNeurons/all_neurons_axons/ion_hip_2um
for f in `ls $input`; do
  echo $f
  awk '$2 == 2 || $2 == 1' ${input}/$f > ${output}/$f
done
