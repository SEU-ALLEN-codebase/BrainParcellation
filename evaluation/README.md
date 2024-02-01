# Reconstruction evaluation
## Data Preview
### Data Description
<div align="center">
   <table frame=void border=0 cellspacing=1>
      <tr>
         <td>
            <p>Total: 103704 (`SWC` after selection)</p>
            <p>Brain involved: 111</p>
            <p>Brain region involved: 281</p>
            <p>Major Brain regions(#>30): 149</p>
         </td>
         <td>
            <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/soma.png" width=350>
         </td>
      </tr>
   </table>
</div>


### Evaluation of dataset
![image](https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/distribution_rough_brains.png)

<div class="1" align='center'>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/CTX_count.png" width=500> <b>   </b>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/HB_count.png" width=500> <b>   </b>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/HY_count.png" width=500> <b>   </b>
</div>
<div clss="2" align='center'>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/CNU_count.png" width=350> <b>   </b>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/TH_count.png" width=350> <b>   </b>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/CB_count.png" width=350> <b>   </b>
   <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/evaluation/figs/MB_count.png" width=450> <b>   </b>
</div>


## Usage
1. Assign the brain region for each neuron (soma), using `soma_reassign.py`
2. Data selection, to filter results with low/over-traced reconstructions, using `filter_neuron.py`
   
