# FSANet
Frequency-Spatial Attention Network with Dual Color Space Fusion for Underwater Image Enhancement
<img width="1328" height="882" alt="image" src="https://github.com/user-attachments/assets/9f01daba-1f0f-4799-968f-237b484005c8" />


## Acknowledgement

This project is built upon the excellent open-source work of **DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement** by Hamidreza Farhadi Tolie, Jinchang Ren, and Eyad Elyan (Robert Gordon University). We are grateful to the authors for making their code publicly available.

- **Code repository:** [https://github.com/hfarhaditolie/DICAM](https://github.com/hfarhaditolie/DICAM)
- **Paper:** Tolie, H. F., Ren, J., & Elyan, E. (2024). DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement. *Neurocomputing*, 584, 127585. [https://doi.org/10.1016/j.neucom.2024.127585](https://doi.org/10.1016/j.neucom.2024.127585)

Our implementation derives parts of its codebase from DICAM, and several utility functions and evaluation routines are adapted from their repository. We thank the authors for their contribution to the underwater image enhancement community.

## Installation

To install FSANet, follow these steps:

1. Clone the repository:

        git clone https://github.com/DWlzm/FSANet/

2. Navigate to the project directory:

        cd FSANet

3. Install dependencies:

        pip3 install torch>=2.0.0 torchvision opencv-python scikit-image numpy pillow tqdm matplotlib pandas thop
