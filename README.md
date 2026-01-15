# DMSRCrack   
# DMSRCrack: A Dual-Encoder Multi-Scale Refinement Network for Robust Crack Segmentation across Diverse Domains

## Overview

### Abstract
Cracks are a critical indicator of structural health in concrete and pavement surfaces. However, accurate segmentation remains challenging due to complex textures, varying lighting conditions, and the diverse morphologies of cracks across different domains. To address these challenges, we propose **DMSRCrack**, a Dual-Encoder Multi-Scale Refinement Network.

Our architecture leverages a dual-branch design: a CNN encoder to capture local texture details and a Transformer encoder to model long-range global dependencies. A novel Multi-Scale Fusion (MSF) module bridges these branches to integrate features effectively, while a Bi-Directional Refinement Head (BRH) ensures precise boundary delineation. Extensive experiments demonstrate that DMSRCrack achieves state-of-the-art performance, particularly in zero-shot cross-domain scenarios, surpassing existing methods in generalization capability.
