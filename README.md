# DMSRCrack   
# DMSRCrack: A Dual-Encoder Multi-Scale Refinement Network for Robust Crack Segmentation across Diverse Domains

## Overview

### Abstract
Cracks are a critical indicator of structural health in concrete and pavement surfaces. However, accurate segmentation remains challenging due to complex textures, varying lighting conditions, and the diverse morphologies of cracks across different domains. To address these challenges, we propose **DMSRCrack**, a Dual-Encoder Multi-Scale Refinement Network.

Our architecture leverages a dual-branch design: a CNN encoder to capture local texture details and a Transformer encoder to model long-range global dependencies. A Multi-Scale Fusion (MSF) module bridges these branches to integrate features effectively, while a Boundary  Refinement Head (BRH) ensures precise boundary delineation. Extensive experiments demonstrate that DMSRCrack achieves state-of-the-art performance, particularly in cross-domain scenarios.
<img width="1341" height="1058" alt="newover" src="https://github.com/user-attachments/assets/d5487b59-bc05-4fdf-a284-64b61d512d8a" />
## 📂 Dataset Download

Due to file size limitations, the processed datasets (DeepCrack, Rissblder, etc.) are hosted on Google Drive:

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Datasets** | ALL dataset that used in the paper | [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1eFRsvghknTze6qdg5FpTy8JMilzlbvGx?usp=sharing) |

## 📂 Resources & Downloads

Due to file size limitations, the processed datasets and the **ready-to-use inference kit** (containing code and weights) are hosted on Google Drive:

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Datasets** | ALL datasets used in the paper (DeepCrack, Rissbilder, etc.) | [Download Datasets](https://drive.google.com/drive/folders/1eFRsvghknTze6qdg5FpTy8JMilzlbvGx?usp=sharing) |
| **Inference Kit** | **Pre-trained weights** + Inference scripts (Plug & Play) | [Download Kit](INSERT_YOUR_DRIVE_LINK_HERE) |

### ⚡ Quick Start with the Inference Kit
The **Inference Kit** linked above allows you to test the model immediately without configuring the full repository.
1. Download and extract the **Inference Kit** zip file.
2. It contains the model weights and the `eval.py` script.
3. Simply place your test images in the folder provided and run the script.
