# CGM-GM
Learning Physics for Unveiling Hidden Earthquake Ground Motions via Conditional Generative Modeling

## Overview


## System Requirements

### Hardware requirements

We train our ``PeRCNN`` and the baseline models on an Nvidia DGX with four Tesla V100 GPU of 32 GB memory. 

### Software requirements

#### OS requirements
 
 - Window 10 Pro
 - Linux: Ubuntu 18.04.3 LTS

#### Python requirements

- Python 3.6.13
- [Pytorch](https://pytorch.org/) 1.6.0
- Numpy 1.16.5
- Matplotlib 3.2.2
- scipy 1.3.1

## Installtion guide

It is recommended to install Python from Anaconda with GPU support, and then install the related packages via conda setting.  

## How to run

### Dataset

Considering the traing data size being over large, we provide a Google drive link for testing our models. Besides, we also uploaded the simulation code with high-order finite difference methods for readers to play with. 

### Implementation

Generally, we evaluate our `PeRCNN` on four tasks: 
- Solving PDEs (compare w/ [PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf))
- Coefficients identification
- Data reconstruction (compare w/ [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), [Recurrent ResNet](https://arxiv.org/pdf/1610.00081.pdf), [PDE-Net](https://arxiv.org/pdf/1710.09668.pdf), [DHPM](https://arxiv.org/pdf/1801.06637.pdf))
- Discovering PDEs (compare w/ [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), [Recurrent ResNet](https://arxiv.org/pdf/1610.00081.pdf), [PDE-Net](https://arxiv.org/pdf/1710.09668.pdf), [DHPM](https://arxiv.org/pdf/1801.06637.pdf))

We present three folders for solving PDEs, data reconstruction and equation discovery. The coefficients identification can be referred to the equation discovery folder, which is essentially the Stage-3 part. More implementation details can be found in each folder. 

## License

This project is released under the GNU General Public License v3.0.

