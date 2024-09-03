# Generation and analyses of CCF-ME atlas
This is the major part of the project, containing the core scripts for microenvironment construction, whole-brain sub-parcellation, and subsequently analyses. 

### Code structure
```
microenviron/
├── intermediate_data/             
│   ├── gen_bihemi_parc.py          # Script generating the bihemisphere atlas from orginal atlas.
│   ├── parc_r671_full.nrrd         # Sub-parcellation atlas based on microenvironments (CCF-ME).
│   └── parc_r671_full.nrrd.pkl     # Mapper from CCF-ME regions to CCF regions.
├── plotters/                       # Plotting utilities.
│   ├── CP_single_morphologies/     # CP-related visualization utilities.
│   ├── feature_separability/       # Feature separability evaluation.
│   ├── hippocampus/                # Visualization of hippocampal microenvironments/single neurons.
│   ├── summary_of_parc/            # Estimation of statistics of CCF-ME atlas.
│   └── whole-brain_projection/     # Utilities for axonal projection characterization.
├── config.py                       # Shared variables and methods for microenvironment analyses.
├── feat_sel.py                     # Methods for Top-K features extraction.
├── generate_me_map.py              # Generation whole-brain/regional microenvironment feature map.
├── get_global_features.py          # Calculation of global features for each neuron.
├── micro_env_features.py           # Construction of microenvironment for each neuron.
├── parcellation.py                 # Sub-parcellation based on microenvironments.
├── preprocess.py                   # Aggregation of additonal features and meta information for neurons.
├── resample_swc.py                 # Resampling neuronal morphologies into homogeneous skeletons.
├── shape_normalize.py              # Shape normalization and sketching of CCF regions.
├── utils.py                        # Miscellaneous utilities.
└── README.md                       # Documentation of the `microenviron` section.
```

### Usage
#### Constructing of microenvironments from neuron morphologies
1. Download the morphologies from NeuroXiv via this https://download.neuroxiv.org. Alternatively, you can use your own dendritic morphologies. Optionally, you may choose to spherically crop the morphologies to ensure they are isotropic. Ensure that the morphologies are aligned in the same standardized space, such as CCFv3.
2. Resample the morphologies (skeletons) to achieve homogeneously spaced skeletons using the `resample_swc.py` script.
3. Calculate the global morphological features (similar to L-Measure) by using `get_global_features.py`.
4. Aggregate additional features and metadata. This can be done with `preprocessing.py`, which will generate the full set of morphological features along with metadata, including brain regions, brain areas, and soma locations.
5. Construct the microenvironments in feature space. First, estimate the appropriate radius (local neighborhood) within which the top 6 neurons are selected to construct the microenvironments using the `estimate_radius` method in `micro_env_features.py`. After determining the radius, calculate the microenvironments using `micro_env_features.py`.

At this point, you should have obtained the microenvironment representations for all neurons.

#### Generating feature map
We have implemented various feature visualization styles, including section-wise 2D feature maps across the entire brain or within specific regions, extraction and plotting of representative morphologies, comparative feature distribution in 3D space, and feature map coloring by clusters. Users can visualize the microenvironment features in any of these styles using `generate_me_map.py`.

#### Generating sub-region atlas based on microenvironments
Sub-parcellation of CCF brain regions was performed using the Leiden community detection algorithm applied to a spatially defined K-nearest-neighbor graph. The edge weights were estimated based on the feature similarity between microenvironments. Detailed methods are provided in our paper. All procedures for generating the atlas are implemented in the script `parcellation.py`.

#### Analyzing and plotting
Exemplar source codes for analyzing and plotting CCF-ME can be found in the corresponding directories under the folder `plotters`. Materials for most figures in our paper are generated using these scripts.

