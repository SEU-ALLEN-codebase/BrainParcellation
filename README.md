# BrainParcellation
This is a comprehensive codebase designed for generating whole brain regional parcellation based on dendritic microenvironments. This repository is maintained mainly by Yufeng Liu (yufeng_liu@seu.edu.cn) from SEU-ALLEN, with the help from Sujun Zhao, Zuo-Han Zhao, and Zhixi Yun.

## Getting Started
### Prerequisites
- Python
- Numpy
- Scipy
- Scikit-learn
- Pandas
- pylib. `pylib` is a customized library developed by Yufeng Liu and Zuo-Han Zhao for manipulating neuronal file parsing, image processing, morphology analysis, anatomical processing, and other specialized utilities for neuron informatics. You can download it from GitHub and add the directory to your `$PYTHONPATH`.
- Other dependencies listed in `requirements.txt`

### Installation
Clone the repository and install the necessary dependencies:
```
git clone https://github.com/SEU-ALLEN-codebase/BrainParcellation.git
cd BrainParcellation
pip install -r requirements.txt
```

### Usage
This repository contains an analytical framework with various tools. You can simply navigate to the specific script and run it according to the instructions provided in the corresponding README file.

### Structure of the project
The structure of the outermost levels of the source code and examples is as follows:
```
BrainParcellation/
│
├── common_lib/              # Common tools and variables
│   └── configs.py                 
├── evaluation/              # Evaluation of the reconstructions
├── microenviron/            # The core scripts, including microenvironment construction and sub-parcellation.
├── reconstruction/          # Source codes and examples for neuron reconstruction.
├── coplanarity/             # Coplanarity of local branches. Deprecated
├── requirements.txt         # List of dependencies
├── README.md                # Project overview and instructions
├── LICENSE                  # License information
└── .gitignore               # Git ignore file
```
For detailed information on each section, please refer to the README.md file located within that section.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgments
Special thanks to all contributors and the SEU-ALLEN team for their support in developing this project.

