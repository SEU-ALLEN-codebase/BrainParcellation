# Please put source codes and examples for local morphology tracing here

### Data Storage
#### Path: /PB/SEU-ALLEN/Users/Sujun/230k_organized_folder
```
|-- 230k_organized_folder/
    |-- app2/                       #app2追踪产生的swc文件     
    |-- v3dpbd/                     #追踪神经元的cropped image blcok(1024*1024*256)          
    |-- mip/                        #图像块的mip
    |-- mip_swc/                    #图像块mip和对应神经元形态结构生成的png图片
    |-- pruned_pre_registered/      #配准前最终重建结果
    |-- 179k_CCFv3_25um_raw/        #CCFv3_25um配准结果                  
    |-- cropped_100um/              #追踪结果配准后的cropped sphere(r=100um)
    |-- filter.py                   #block图像增强处理
    |-- filter-job.sh
    |-- recon-job.sh                #run app2
    |-- soma_finder.py              #提取soma
    |-- prune-job.sh                #追踪生成的swc做结构pruning（去除短分支、去除多分叉、去除错误角度的分支）
    |-- seg_prune.py           
    |-- supp.csv                    #brain info
    |-- data_info.txt               #readme
    |-- analysis
        |-- gf_179k_crop.csv        # L-measure(22 features) table for r=100um cropped sphere
        |-- gf_1891_crop.csv
```


### Characteristic Distribution compared with 1876(gold standard) dataset
#### Pearson correlation between whole faeture matrix(22 features) : 0.56
![image](https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/data/reconstruction/figures/1891_comparison.png)
