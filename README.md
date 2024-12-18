# Hyperspectral Zoo

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


## Description

This a testing ground for implementing deep learning models for hyperspectral data.

## Datasets

Datasets downloaded from [here](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).

 | Dataset | # Bands | Image Size  |# Classes | Sensor (spectral range) |
 | --- | --- | --- | --- | --- |
 | Indian Pines | 200 | 145 x 145 | 16 | AVIRIS ($0.4 - 2.5 * 10^{-6}$ m)|
 | Salinas | 204 | 512 x 217 | 16 | AVIRIS |
 | Pavia Centre | 102 | 1096 x 1096 | 10 | ROSIS |
 | Pavia University | 103 | 610 x 610 | 10 | ROSIS |
 | Kennedy Space Center | 224 | 512 x 614 | 13 | AVIRIS |
 | Botswana | 145 | | 14 | NASA EO-1 ($0.4 - 2.5 * 10^{-6} m$) |

## Models
- [1D CNN](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2015/258619) -  Hu et al. "Deep Convolutional Neural Networks for Hyperspectral Image Classification", Journal of Sensors, 2015
- [HSI-CNN](https://arxiv.org/pdf/1802.10478) -  Luo et al. "HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image", ICALIP 2018
- [Multi-Scale CNN](https://ieeexplore.ieee.org/document/8297014) He et al. "Multi-scale 3D deep convolutional neural network for hyperspectral image classification", IEEE Conference on Image Processing, 2017
- [Spectral-Spatial CNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565) - Ben Hamida et al. "3-D Deep Lear Approach for Remote Sensing Image Classification", IEEE TGRS, 2018

## Set-Up

#### Conda

```bash
# clone project
git clone https://github.com/abigailstone/Hyperspectral-Zoo
cd Hyperspectral-Zoo

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

Log in to [Weights & Biases](https://wandb.ai/) from the command line to set up W&B logging.

## Training

Train model with default configuration

```bash
# train on CPU
python src/train.py

# train using Salinas data
python src/train.py data=salinas

```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

See [Hydra docs](https://hydra.cc/) for more info.
