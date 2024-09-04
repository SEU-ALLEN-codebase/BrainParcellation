# Reconstruction evaluation
## Data
### Data summary
<div align="center">
   <table frame=void border=0 cellspacing=1>
      <tr>
         <td>
            <p>Total: 101,136</p>
            <p>Brain involved: 111</p>
            <p>Brain region involved: 528</p>
            <p>Brain regions (#>10): 397</p>
         </td>
         <td>
            <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/soma.png" width=350>
         </td>
      </tr>
   </table>
</div>


### Evaluation of dataset
Numbers of the reconstructions:
- 182,497 (182k): The total number of cell bodies (somas) utilized in this work. Please refer to the paper: https://www.researchsquare.com/article/rs-3146034/v1.
- 179,568 (179k): The total number of neurons reconstructed successfully.
- 101,136 (101k): The neurons after filtering. Filtering criteria are: (1) A good reconstruction should be located in the 582 non-fiber-tract, non-ventricle salient CCF regions; (2) A good reconstruction should have morphological features similar to manually annotated morphologies.
- 103,603 (103k): Similar to the 101k set, but keeps the neurons within 50 μm of salient regions.

For more information, please refer to our paper: **To be updated when the manuscript submitted**

### Code structure
```
evaluation/
│
├── code/                           # Source codes.
│   ├── data_summary.py             # Evaluation of distributions and qualities.
│   ├── filter_neuron.py            # Filtering of the neurons based on soma locations and global features.
│   ├── match_gs_recon.py           # Finding the correspondence between SEU-A1876 and 101k.
│   └── soma_reassign.py            # Reassign the brain regions for neurons not in 101k but within 50 μm of salient regions.
├── data/                           # Exemplar meta information for the reconstructed neurons.
│   ├── 179k_soma_region.csv        # The CCF brain regions for the 179k reconstructed neurons.
│   ├── 179k_somalist.txt           # The coordinates (in μm) of somas for the 179k neurons.
│   ├── 1876_soma_region.csv        # Meta information for the manually reconstructed SEU-A1876 neurons.
│   ├── filtered_soma.txt           # The coordinates (in 25 μm) of somas for the 103k neurons.
│   ├── final_filtered_swc.txt      # Name list of the 103k neurons.
│   ├── gf_1876_crop.csv            # Global features of the 1,876 annotated neurons.
│   ├── region_match_table.csv      # The mapping of CCF brain regions to the CCF brain areas/structures.
│   └── so_match_table.txt          # The correspondence between 103k and SEU-A1876.
├── figs/                           # Figures showing the distribution of neurons across brain areas
└── README.md                       # Instruction of this section
```




