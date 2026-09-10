# LWNNet-BCD

## An Explainable Lightweight Neural Network for Breast Cancer Detection in 2D Ultrasound Images Using Feature Mapping

This repository provides the implementation and reproducibility materials for **LWNNet-BCD**, a compact multi-task neural framework for breast ultrasound analysis.

LWNNet-BCD integrates:

- lesion segmentation,
- boundary localization,
- hierarchical lesion-aware feature mapping,
- Normal/Benign/Malignant diagnostic prediction,
- Grad-CAM-based model localization,
- deletion-based faithfulness evaluation, and
- independent external validation.

The framework was developed using the **Breast Ultrasound Images dataset (BUSI)** and independently evaluated on the **BrEaST** and **BUS-UCLM** datasets.

> **Important:** The term *lightweight* in this work refers to reduced parameter count, model-storage requirement, and arithmetic complexity. It does not imply superior GPU inference speed.

---

## Repository

Official repository:

https://github.com/vamsib111/LWNNet-BCD

---

## Authors

- **Bandi Vamsi**
- **Nithin Sai**
- **Ali Al Bataineh**
- **Bhanu Prakash Doppala**

---


## Repository Structure

The repository is organized as follows:

```text
LWNNet-BCD/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── losses.py
│   ├── train.py
│   ├── inference.py
│   └── gradcam.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate_internal.py
│   ├── evaluate_external.py
│   └── bootstrap_analysis.py
│
├── configs/
│   └── lwnnet_bcd_config.yaml
│
├── splits/
│   └── README.md
│
├── checkpoints/
│   └── README.md
│
├── examples/
│   └── README.md
│
└── docs/
    └── reproducibility.md

Datasets
BUSI

BUSI was used for model development and final internal evaluation.

Original dataset composition:

Normal: 133 images
Benign: 437 images
Malignant: 210 images
Total: 780 images

Following the integrity audit and removal of 11 duplicate or otherwise problematic cases:

Eligible images: 769
Duplicate-aware leakage-control groups: 684

BrEaST: BrEaST was used only for independent external validation.

Total scans: 256
Patients: 256
Normal: 4
Benign: 154
Malignant: 98

External lesion segmentation used the 252 lesion-containing scans.

Dataset identifier: DOI: 10.7937/9WKK-Q141

BUS-UCLM: BUS-UCLM Version 3 was used only for independent external validation.

Total images: 683
Normal: 419
Benign: 174
Malignant: 90

External lesion segmentation used 260 images with compatible lesion annotations.

Additional sensitivity analyses included:

strict B-mode-like subset: 521 images;
exact-deduplicated subset: 679 images.

Dataset identifier: DOI: 10.17632/7fvgj4jsp7.3

keep the final BUSI holdout isolated from model development;
do not optimize thresholds or model settings using external datasets.

The final holdout and external datasets must not be used for architecture selection, hyperparameter tuning, or post hoc calibration if the reported protocol is to be reproduced faithfully.

