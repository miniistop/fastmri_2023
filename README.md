# 2023 SNU FastMRI Challenge

## Overview
This project implements an advanced MRI reconstruction pipeline that combines end-to-end variational networks with cross-domain feature fusion. The model combines an E2E-VarNet backbone with a novel cross-domain fusion network to achieve high-quality MRI reconstruction from undersampled k-space data.

### Architecture Overview
![Overall Architecture](/docs/images/overall_architecture.png)

Our full pipeline architecture consists of two main modules (Module 1 and Module 2) as shown below:

1. **Module 1: E2E-VarNet Backbone**
![Module 1: E2E-VarNet Backbone](/docs/images/module1_e2e-varnet_backbone.png) ![Module 1: Attention VarNet](/docs/images/module1_attention_varnet.png)
   - Input: K-space data
   - Output: Initial image reconstruction
   - Components:
     - Refinement (U-Net → Attention U-Net)
     - Data consistency
     - Sensitivity map estimation (U-Net)

2. **Module 2: Cross-Domain Feature Fusion Network (CDFFNet)**
![Module 2: Cross-Domain Feature Fusion Network](/docs/images/module2_cross-domain_feature_fusion_network.png)
   - Custom cross-domain network
   - K-Net: MWCNN
   - I-Net: Attention U-Net (6 layers)
   - Fusion: Softmax-based weighted strategy

## Directory Structure
```bash
├── FastMRI_challenge/  # This repository
├── Data/
│   ├── train/
│   │   ├── image/
│   │   └── kspace/
│   ├── val/
│   │   ├── image/
│   │   └── kspace/
│   └── leaderboard/
│       ├── acc4/
│       └── acc8/
└── result/
    └── [model_name]/
        ├── checkpoints/
        ├── reconstructions_val/
        └── reconstructions_leaderboard/
```

### Data Format
- Training/Validation files: `brain_{mask_type}_{number}.h5`
  - mask_type: "acc4" or "acc8"
  - acc4 numbers: 1-203
  - acc8 numbers: 1-204
- Leaderboard files: `brain_test_{number}.h5` (numbers 1-58)

## Learning Strategy

### Data Augmentation
![Data Augmentation](/docs/images/data_augmentation.png)
1. Random Data Transformation
   - Mask shifting (0~7)
2. 2x Data Augmentation
   - Acc4 ↔ Acc8 mask change
   - Expanded from 5,674 to 11,348 slices

### Training Techniques
- Decoupled learning & Freezing modules: E2E VarNet → CDFFNet
- Learning Rate: ReduceLROnPlateau scheduler
- Optimizer: AdamW
- Gradient accumulation
- Hyperparameter tuning

## Installation and Setup

### Requirements
Python 3.8.10
```bash
pip install torch
pip install numpy
pip install requests
pip install tqdm
pip install h5py
pip install scikit-image
pip install pyyaml
pip install opencv-python
pip install matplotlib
```

### Usage

1. **Training**
```bash
python train.py
```
- Saves validation reconstructions to `result/reconstructions_val/`
- Records per-epoch validation loss

2. **Reconstruction**
```bash
python reconstruct.py
```
- Generates leaderboard reconstructions in `result/reconstructions_leaderboard/`

3. **Evaluation**
```bash
python leaderboard_eval.py
```
- Calculates SSIM values for both 4X and 8X sampling masks

## References
1. Chen, L., et al. "Simple baselines for image restoration." ECCV 2022
2. Chen, Y., et al. "AI-based reconstruction for fast MRI—A systematic review and meta-analysis." Proceedings of the IEEE 110.2 (2022)
3. Oktay, O., et al. "Attention u-net: Learning where to look for the pancreas." arXiv:1804.03999 (2018)
4. Sriram, A., et al. "End-to-end variational networks for accelerated MRI reconstruction." MICCAI 2020

## Submission Package
- GitHub repository with detailed execution instructions
- Loss graphs and training records
- Model weight files
- Model explanation presentation

















### Overall Architecture
![Overall Architecture](./docs/images/overall_architecture.png)

Our complete pipeline architecture consists of two primary modules that work in sequence to achieve optimal MRI reconstruction:

## Our Model

### Module 1: E2E-VarNet Backbone
Module 1 serves as the backbone of our architecture, utilizing an End-to-End Variational Network approach.

#### Architecture Components:
![Module 1 Details](./docs/images/module1_details.png)

1. **Base Architecture**
   - Input: K-space data
   - Output: Initial image reconstruction
   - Key Components:
     - Data Consistency (DC & R)
     - Sensitivity Map Estimation (SME)
     - Inverse Fourier Transform (IFT)

2. **Attention U-Net Implementation**
![Attention U-Net Architecture](./docs/images/module1_attention_varnet.png)

   - Refinement module evolution: U-Net → Attention U-Net (4 layers)
   - Enhanced feature attention mechanism
   - Down/Up layers with precise specifications:
     - Down Layer: (3x3 Conv. + BN + LeakyReLU + DropOut) (x2)
     - Up Layer: (3x3 Conv. + BN + LeakyReLU + DropOut) (x2) + 1X1 Conv.

### Module 2: Cross-Domain Feature Fusion Network (CDFFNet)
![CDFFNet Architecture](./assets/images/cdf_architecture.png)

Our custom cross-domain network incorporates:
- K-Net: Multi-Level Wavelet CNN (MWCNN)
- I-Net: Attention U-Net (6 layers)
- Fusion: Softmax-based weighted strategy
- Dual-domain processing with FT/IFT operations

## Learning Strategy

### Data Augmentation
![Data Augmentation Strategy](./assets/images/data_augmentation.png)

1. **Random Data Transformation**
   - Mask shifting (range: 0-7)
   - Enhanced sampling diversity

2. **2x Data Augmentation**
   - Acc4 ↔ Acc8 mask interchange
   - Dataset expansion: 5,674 → 11,348 slices

### Training Techniques
1. **Advanced Learning Strategy**
   - Decoupled learning & Freezing modules: E2E VarNet → CDFFNet
   - Learning Rate: ReduceLROnPlateau scheduler
   - Optimizer: AdamW
   - Gradient accumulation
   
2. **Data Handling Optimizations**
   - Adjusting data composition (Acc8:Acc4 = 6:4)
   - 2x gradient for challenging tasks (Slice index #0-#3; eye level)
   - K-space data structure unification for batch learning (# of Coil = 20)



### Architecture Overview
![Overall Architecture](path/to/architecture.png)

Our solution consists of two main modules:

1. **Module 1: E2E-VarNet Backbone**
   - Input: K-space data
   - Output: Initial image reconstruction
   - Components:
     - Refinement (U-Net → Attention U-Net)
     - Data consistency
     - Sensitivity map estimation (U-Net)

2. **Module 2: Cross-Domain Feature Fusion Network (CDFFNet)**
   - Custom cross-domain network
   - K-Net: MWCNN
   - I-Net: Attention U-Net (6 layers)
   - Fusion: Softmax-based weighted strategy

## Directory Structure
```bash
├── FastMRI_challenge/  # This repository
├── Data/
│   ├── train/
│   │   ├── image/
│   │   └── kspace/
│   ├── val/
│   │   ├── image/
│   │   └── kspace/
│   └── leaderboard/
│       ├── acc4/
│       └── acc8/
└── result/
    └── [model_name]/
        ├── checkpoints/
        ├── reconstructions_val/
        └── reconstructions_leaderboard/
```

### Data Format
- Training/Validation files: `brain_{mask_type}_{number}.h5`
  - mask_type: "acc4" or "acc8"
  - acc4 numbers: 1-203
  - acc8 numbers: 1-204
- Leaderboard files: `brain_test_{number}.h5` (numbers 1-58)

## Learning Strategy

### Data Augmentation
1. Random Data Transformation
   - Mask shifting (0~7)
2. 2x Data Augmentation
   - Acc4 ↔ Acc8 mask change
   - Expanded from 5,674 to 11,348 slices

### Training Techniques
- Decoupled learning & Freezing modules: E2E VarNet → CDFFNet
- Learning Rate: ReduceLROnPlateau scheduler
- Optimizer: AdamW
- Gradient accumulation
- Hyperparameter tuning

## Installation and Setup

### Requirements
Python 3.8.10
```bash
pip install torch
pip install numpy
pip install requests
pip install tqdm
pip install h5py
pip install scikit-image
pip install pyyaml
pip install opencv-python
pip install matplotlib
```

### Usage

1. **Training**
```bash
python train.py
```
- Saves validation reconstructions to `result/reconstructions_val/`
- Records per-epoch validation loss

2. **Reconstruction**
```bash
python reconstruct.py
```
- Generates leaderboard reconstructions in `result/reconstructions_leaderboard/`

3. **Evaluation**
```bash
python leaderboard_eval.py
```
- Calculates SSIM values for both 4X and 8X sampling masks

## References
1. Chen, L., et al. "Simple baselines for image restoration." ECCV 2022
2. Chen, Y., et al. "AI-based reconstruction for fast MRI—A systematic review and meta-analysis." Proceedings of the IEEE 110.2 (2022)
3. Oktay, O., et al. "Attention u-net: Learning where to look for the pancreas." arXiv:1804.03999 (2018)
4. Sriram, A., et al. "End-to-end variational networks for accelerated MRI reconstruction." MICCAI 2020

## Submission Package
- GitHub repository with detailed execution instructions
- Loss graphs and training records
- Model weight files
- Model explanation presentation