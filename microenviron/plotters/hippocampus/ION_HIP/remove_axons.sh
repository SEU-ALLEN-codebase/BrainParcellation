#for fpath in `ls /PBshare/SEU-ALLEN/Users/Sujun/ION_Hip_CCFv3_crop/*swc`; do fn=$(basename $fpath); awk '$2 != 2' $fpath > ./swc_dendrites/${fn}; echo $fn; done

for fpath in `ls /PBshare/SEU-ALLEN/Users/Sujun/ION_Hip_CCFv3_crop/*swc`; do fn=$(basename $fpath); awk '$2 == 1 || $2 == 2' $fpath > ./swc_axons/${fn}; echo $fn; done
