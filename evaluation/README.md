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
         </td>
         <td>
            <img src="https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/data/evaluation/figs/soma.png" width=350>
         </td>
      </tr>
   </table>
</div>

### Evaluation of dataset
![image](https://github.com/SEU-ALLEN-codebase/BrainParcellation/blob/main/data/evaluation/figs/distribution_over_brain_regions.png)

## Usage
1. Assign the brain region for each neuron (soma), using `soma_reassign.py`
2. Data selection, to filter results with low/over-traced reconstructions, using `filter_neuron.py`
   
