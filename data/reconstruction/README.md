# Please put source codes and examples for local morphology tracing here

##Data
####Path: /PB/SEU-ALLEN/Users/Sujun/230k_organized_folder
/PB/SEU-ALLEN/Users/Sujun/230k_organized_folder
|-- 230k_organized_folder/
    |-- app2/                       #app2追踪产生的swc文件     
    |-- v3dpbd/                     #追踪神经元的cropped image blcok(1024*1024*256)          
    |-- mip/                        #图像块的mip
    |-- mip_swc/                    #图像块mip和对应神经元形态结构生成的png图片
    |--                             
    |-- cropped_100um/              #追踪结果的cropped sphere(r=100um)
    |-- filter.py                   #block图像增强处理
    |-- filter-job.sh
    |-- recon-job.sh                #run app2
    |-- soma_finder.py              #提取soma
    |-- prune-job.sh                #追踪生成的swc做结构pruning（去除短分支、去除多分叉、去除错误角度的分支）
    |-- seg_prune.py           
    |-- supp.csv                    #brain info
    |-- data_info.txt               #readme

