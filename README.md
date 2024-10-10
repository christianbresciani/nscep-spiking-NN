# nscep-spyking-NN

This repository hosts the code used in the following thesys:

- Christian Bresciani, Approaches to human activity recognition via passive radar

The code is provided under an MIT license.
Please consider citing the above works if you use the code in this repository.

## Source code

There are several Python scripts and Jupyter notebooks in this repository.

- **NeuroSpykeHAR.ipynb** contains a compact version to run the downsampling, training and testing of the different networks;
- **SNN/neural.py** contains the code used to train and test the basic CNN and SNN;
- **SNN/networks.py** contains the basic CNN and SNN architectures;
- **SNN/nsTrain.py** contains the code used to train and test NeuroSpykeHAR, a neuro-symbolic architecture with SNN for human activity recognition using Wi-Fi sensing data;
- **SNN/nsNetworks.py** contains the CNN and SNN architectures of the neural part;
- **SNN/logic.pl** contains the DeepProbLog code of NeuroSpykeHAR;
- **SNN/nsTempTrain.py** contains the code used to train and test version of NeuroSpykeHAR with a temporal logic;
- **SNN/logic.pl** contains the DeepProbLog code of NeuroSpykeHAR with temporal logic;
- **SNN/results.py** contains the code used confront NeuroSpykeHAR and DeepProbHAR;
- **SNN/deepprobhar.py** contains the code used to train and test DeepProbHAR, a neuro-symbolic architecture for human activity recognition using Wi-Fi sensing data (see https://github.com/marcocominelli/csi-vae/tree/fusion2024);
- **SNN/har.pl** contains the DeepProbLog code of DeepProbHAR;

- **features-2d-time-split.ipynb** and **rule-based-classification** contain the video analysis for the extraction of the rules for the DeepProbLog codes

**How to run the code:**
It is advised to install all the required packages in a new Conda environment:
```
  $ conda create --name nscepspyking python=3.12
  $ conda activate nscepspyking
```

Then, you can use `pip` to install the required packages:
```
  $ pip install -r requirements.txt
```

The code has been tested on an Ubuntu 22 distribution using Python 3.12.3.

## Other useful links

- [Raw CSI dataset](https://doi.org/10.5281/zenodo.7732595)