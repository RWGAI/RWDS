<div align="center">
  <img src="https://qcriscairw-bcf4f2eec0-endpoint.azureedge.net/blobqcriscairwbcf4f2eec0/wp-content/uploads/2025/03/new_logo_v3_hori.png" alt="RWDS Logo" width="800"/>
</div>

<div align="center">

# Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery
### **Real-World Distribution Shifts (RWDS) Benchmark**

</div>

<div align="center">
  
📄 [**CVPR 2025**](https://openaccess.thecvf.com/content/CVPR2025/html/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.html) | 🔗 [**Paper**](https://arxiv.org/abs/2503.19202) | 🎥 [**Video**](https://www.youtube.com/watch?v=_sZjkwVUSow) | 📋 [**Poster**](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202025/32546.png?t=1748807138.4285972) | 🌐 [**Project Page**](https://rwgai.com/rwds/) | 💻 [**Code**](#)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

### Abstract

Object detectors have achieved remarkable performance in many applications; however, these deep learning models are typically designed under the i.i.d. assumption, meaning they are trained and evaluated on data sampled from the same (source) distribution. In real-world deployment, however, target distributions often differ from source data, leading to substantial performance degradation. Domain Generalisation (DG) seeks to bridge this gap by enabling models to generalise to Out-Of-Distribution (OOD) data without access to target distributions during training, enhancing robustness to unseen conditions. In this work, we examine the generalisability and robustness of state-of-the-art object detectors under real-world distribution shifts, focusing particularly on spatial domain shifts. Despite the need, a standardised benchmark dataset specifically designed for assessing object detection under realistic DG scenarios is currently lacking. To address this, we introduce Real-World Distribution Shifts (RWDS), a suite of three novel DG benchmarking datasets that focus on humanitarian and climate change applications. These datasets enable the investigation of domain shifts across (i) climate zones and (ii) various disasters and geographic regions. To our knowledge, these are the first DG benchmarking datasets tailored for object detection in real-world, high-impact contexts. We aim for these datasets to serve as valuable resources for evaluating the robustness and generalisation of future object detection models. 

🛰️ **TL;DR:** RWDS is a comprehensive suite of benchmarks for evaluating the robustness of object detection models under real-world distribution shifts in satellite imagery when conditions change in real-world scenarios. It includes three datasets focused on disaster assessment and climate applications.

## 📊 Dataset Download

**The RWDS datasets are available on both HuggingFace and Zenodo:**

### 🤗 HuggingFace Hub
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/RWGAI/RWDS)

**Direct download:**
```bash
# Using HuggingFace Hub
pip install huggingface_hub
huggingface-cli download RWGAI/RWDS --repo-type dataset
```

### 🗂️ Zenodo Archive
[![DOI](https://zenodo.org/badge/DOI/[PLACEHOLDER_DOI].svg)](https://doi.org/[PLACEHOLDER_DOI])

**Direct download link:** `[PLACEHOLDER - INSERT ZENODO LINK HERE]`

## 📁 Installation and Setup

### 1️⃣ Framework Requirements

**Download MMDetection Framework:**
Install MMDetection from the official repository by following the [official MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).
<!---
\> **Note:** *We may release our customized version of MMDetection optimized for this benchmark in the future.*
-->
### 2️⃣ Directory Structure Setup
```bash
# Recommended directory structure:
your_project/
├── mmdetection/                   # Official MMDetection installation
│   └── configs/                   # MMDetection config directory
└── RWDS_Dataset/                 # RWDS datasets (outside mmdetection)
    ├── RWDS-CZ/
    ├── RWDS-FR/
    └── RWDS-HE/
```

### 3️⃣ Configuration Setup
After installing MMDetection, copy the RWDS configuration files:

```bash
# Copy RWDS config files to MMDetection
cp -r RWDS_mmdetection/configs/* mmdetection/configs/
```

⚠️ **Important:** 
- Place `RWDS_Dataset` directory **outside** the `mmdetection` directory
- Copy all files from `RWDS_Dataset/configs/` to `mmdetection/configs/` upon installation
- Verify the directory structure matches the recommended layout above

## 🔬 Experiments

<!--- For detailed training and evaluation instructions, see the respective README files:

- 🌦️ [RWDS-CZ/README.md](RWDS-CZ/README.md)
- 🌊 [RWDS-FR/README.md](RWDS-FR/README.md)
- 🌀 [RWDS-HE/README.md](RWDS-HE/README.md) -->

> **Stay-tuned:** *We willl release detailed training and evaluation instructions which we used in RWDS soon.*

## 📝 Citation

```bibtex
@inproceedings{alemadi2025rwds,
  title={Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery},
  author={Al-Emadi, Sara A. and Yang, Yin and Ofli, Ferda},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---
*💻 Code and detailed documentation coming soon.*
