# CGM-GM
Learning Physics for Unveiling Hidden Earthquake Ground Motions via Conditional Generative Modeling

Paper link: [arXiv](https://arxiv.org/abs/2407.15089)

---

## Overview

This work addresses a critical problem in seismic hazard assessment and infrastructure resilience: predicting high-fidelity ground motions for future earthquakes. We propose a new AI simulator, Conditional Generative Modeling for Ground Motion (CGM-GM), to synthesize high-frequency and spatially continuous earthquake ground motion waveforms. CGM-GM uses earthquake magnitudes and geographic coordinates of earthquakes and sensors as inputs.



### Highlights

- **Learning physics**: CGM-GM can capture the underlying spatial heterogeneity and physical characteristics. 

- **Comprehensive evaluations**: The framework provides comprehensive evaluations in time and frequency domains.  

- **Great potential**: CGM-GM demonstrates a strong potential for outperforming a state-of-the-art non-ergodic empirical ground motion model and shows great promise in seismology and beyond.

Below is an example of generated FAS maps in the San Francisco Bay Area.

<p align="center">
    <img width=85%" src="asset/fas_maps.png">
</p>

---

## System requirements and installation

### Hardware requirements
 
All the experiments are performed on an NVIDIA A100 Tensor Core GPU of 40GB memory.

### OS requirements

All the experiments are performed on SUSE Linux Enterprise Server 15 SP5.

### Python requirements

Install the required dependencies:
```shell
conda create -n cgm_gm python=3.9.17
conda activate cgm_gm
pip install -r requirements.txt
```

---

## Usage

### Dataset

The earthquake dataset in the SFBA was originally downloaded from [NCEDC](https://ncedc.org/). The training and testing dataset in this study is preprocessed and can be found in a data report, which will be made public soon.

### Implementations

1. Train the CGM-GM with Hyperopt for hyper-parameters optimization:

    ```python
    python train_hyperopt.py

    # if training CGM-baseline (with rupture distances)
    python train_hyperopt.py --tcondvar 4
    ```

2. Evaluate the best model:

    ```python
    python test_best_model.py
    ```

3. Generate waveforms from a 100x100 grid:

    ```python
    python generate_points.py
    ```

The implementations of ergodic and non-ergodic GMM for the SFBA can be found in [this paper](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/113/5/2144/623913/Methodology-for-Including-Path-Effects-Due-to-3D?redirectedFrom=fulltext). 

The evaluations include the comparisons of waveform shapes, P and S arrival time, amplitude spectra, and Fourier amplitude spectra maps. We use the [PhaseNet](https://github.com/AI4EPS/PhaseNet) to pick the arrival time of P and S waves. 




## License

This project is released under the GNU General Public License v3.0.

