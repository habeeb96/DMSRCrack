# DMSRCrack   
# DMSRCrack: A Dual-Encoder Multi-Scale Refinement Network for Robust Crack Segmentation across Diverse Domains

## Overview

### Abstract
Cracks are a critical indicator of structural health in concrete and pavement surfaces. However, accurate segmentation remains challenging due to complex textures, varying lighting conditions, and the diverse morphologies of cracks across different domains. To address these challenges, we propose **DMSRCrack**, a Dual-Encoder Multi-Scale Refinement Network.

Our architecture leverages a dual-branch design: a CNN encoder to capture local texture details and a Transformer encoder to model long-range global dependencies. A Multi-Scale Fusion (MSF) module bridges these branches to integrate features effectively, while a Boundary  Refinement Head (BRH) ensures precise boundary delineation. Extensive experiments demonstrate that DMSRCrack achieves state-of-the-art performance, particularly in cross-domain scenarios.
<img width="1341" height="1058" alt="newover" src="arch.png" />

## 📂 Dataset Download

Due to file size limitations, the processed datasets (DeepCrack, Rissblder, etc.) are hosted on Google Drive:

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Datasets** | ALL dataset that used in the paper | [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1eFRsvghknTze6qdg5FpTy8JMilzlbvGx?usp=sharing) |

## 🧪 Inference & Weights

We provide a standalone **Test Kit** on Google Drive containing the inference code (`Inference.py`) and all pre-trained weights. This allows you to run the model immediately without setting up the full repository.

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Test Kit** | Inference Code + All Weights | [Download Test Kit](https://drive.google.com/drive/folders/1BI9O2GaCg_HqHYwHNqkgU_dVn6ZhgPSc?usp=sharing) |
