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

## Overview

LWNNet-BCD is designed as a compact multi-task neural network in which lesion localization and diagnostic prediction are learned within the same framework.

For an input breast ultrasound image, the model:

1. extracts contextual features using a lightweight shared encoder;
2. generates lesion and boundary probability maps;
3. constructs:
   - a global diagnostic descriptor,
   - a lesion-weighted diagnostic descriptor, and
   - a boundary-weighted diagnostic descriptor;
4. combines these descriptors using an adaptive evidence-gating mechanism; and
5. performs hierarchical Normal/Benign/Malignant prediction by modeling:
   - lesion presence, and
   - conditional malignancy.

Ground-truth masks are used only during training. At inference time, the model requires only the breast ultrasound image.

---

## Main Contributions

The implementation supports the following components of the study:

1. **Compact multi-task architecture** integrating lesion segmentation, boundary localization, and hierarchical diagnosis.

2. **Hierarchical lesion-aware feature mapping** using global, lesion-weighted, and boundary-weighted diagnostic evidence.

3. **Adaptive evidence fusion** for constructing the final diagnostic representation.

4. **Controlled BUSI evaluation** using duplicate-aware development folds and a protected one-time final holdout.

5. **Matched diagnostic comparisons** with VGG16 and conventional machine-learning classifiers.

6. **Component ablation** comparing the complete lesion-aware model with a global-only diagnostic variant.

7. **Quantitative Grad-CAM evaluation** using localization and deletion-based faithfulness measures.

8. **Independent external validation** on BrEaST and BUS-UCLM without model retraining or threshold optimization.

---

## Model Architecture

The main architectural configuration is:

| Component | Configuration |
|---|---|
| Input | RGB ultrasound image, `3 × 128 × 128` |
| Shared encoder output | `192 × 16 × 16` |
| Localization feature map | `64 × 128 × 128` |
| Lesion head | Single-channel lesion logit |
| Boundary head | Single-channel boundary logit |
| Diagnostic adapter | `1 × 1` convolution, `192 → 96`, GroupNorm, SiLU |
| Diagnostic feature map | `96 × 16 × 16` |
| Global descriptor | 96-D |
| Lesion-weighted descriptor | 96-D |
| Boundary-weighted descriptor | 96-D |
| Concatenated descriptor | 288-D |
| Evidence gate | `288 → 48 → 3`, softmax |
| Fused diagnostic descriptor | 96-D |
| Diagnostic heads | Lesion presence + conditional malignancy |
| Trainable parameters | **210,684** |

The final three-class probabilities are obtained hierarchically as:

- `P(Normal) = 1 - qL`
- `P(Benign) = qL × (1 - qC)`
- `P(Malignant) = qL × qC`

where `qL` represents lesion-presence probability and `qC` represents conditional malignancy probability.

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

Preprocessing

The frozen preprocessing pipeline consists of:

conversion to RGB;
resizing images to 128 × 128 using area-based interpolation;
resizing masks using nearest-neighbor interpolation;
3 × 3 median filtering;
3 × 3 Gaussian filtering with sigma = 0;
min-max image normalization.

Training augmentation consists of:

horizontal flipping with probability 0.5;
random rotation between -15° and +15°.

No augmentation is applied during validation, fold evaluation, final holdout evaluation, or external validation.

No dataset-specific enhancement, test-time augmentation, performance-guided preprocessing, or post hoc filtering is used during final or external evaluation.

Training Configuration

The principal frozen training configuration is:

Setting	Value
Optimizer	AdamW
Initial learning rate	0.001
Weight decay	0.0001
Batch size	16
Maximum development epochs	80
Random seed	20260830
Fold-specific hyperparameter search	None
Deterministic execution	Enabled

The final model was trained on all 614 BUSI development images for 26 epochs.

Learning-rate schedule:

Epochs 1–23: 0.001
Epochs 24–26: 0.0005

No separate validation partition or early stopping was used during final full-development training.

Loss Function

LWNNet-BCD is optimized using a joint objective consisting of:

lesion segmentation loss,
boundary localization loss, and
hierarchical diagnostic loss.

The complete objective is:

L_total = L_segmentation
        + 0.25 × L_boundary
        + 0.50 × L_diagnosis

Lesion and boundary objectives combine weighted binary cross-entropy and soft Dice loss.

The diagnostic objective combines:

lesion-presence classification, and
conditional malignancy classification.

All loss weights and class weights were fixed before final holdout and external evaluation.

Evaluation Protocol
BUSI Development

The 614-image development cohort was evaluated using a duplicate-aware five-fold protocol.

No leakage-control group was shared between training, validation, and fold-evaluation partitions.

Development results are reported as the unweighted mean and sample standard deviation across the five folds.

Final BUSI Holdout

The 155-image final holdout was evaluated once after freezing:

architecture,
preprocessing,
losses,
training configuration,
decision thresholds, and
evaluation protocol.

No retraining, calibration, threshold modification, or additional model selection was performed after final holdout access.

Final BUSI Holdout Results
Task	Metric	Result
Lesion segmentation	Dice	0.7497
Lesion segmentation	IoU	0.6374
Lesion segmentation	Precision	0.8239
Lesion segmentation	Recall	0.7410
Three-class diagnosis	Accuracy	0.7161
Three-class diagnosis	Balanced accuracy	0.7769
Three-class diagnosis	Macro-F1	0.7077
Three-class diagnosis	Macro ROC-AUC	0.8996
Cancer detection	Sensitivity	0.7857
Cancer detection	Specificity	0.9204
Cancer detection	F1-score	0.7857
Cancer detection	ROC-AUC	0.9216
Cancer detection	Average precision	0.8097

BUSI segmentation values are case-level means across all 155 holdout images under the prespecified normal-target handling.

Model Compactness

LWNNet-BCD contains 210,684 trainable parameters.

Comparison with the VGG16 reference:

Measure	LWNNet-BCD	VGG16
Parameters	210,684	14,716,227
State-dictionary size	0.832 MiB	56.148 MiB
MACs	336.07 M	5.01 B
Estimated FLOPs	672.14 M	10.02 B

Relative reductions:

Parameters: 98.57%
Model storage: 98.52%
MACs: 93.29%

The compact architecture did not produce faster GPU inference in the tested NVIDIA GeForce RTX 3060 environment.

Therefore, the term lightweight in this project refers to architectural compactness, storage requirement, and arithmetic complexity rather than inference-speed superiority.

Explainability Evaluation

Grad-CAM is applied directly to the native neural diagnostic branch.

Quantitative analysis includes:

lesion energy fraction;
pointing-game accuracy;
saliency contrast;
pixel-level ROC-AUC;
deletion-based faithfulness.

Across all 614 development images, removal of Grad-CAM-selected evidence produced a mean diagnostic probability drop of:

0.3110

compared with:

0.1137

for matched random deletion.

The resulting mean faithfulness advantage was:

0.1973

with a 95% group-bootstrap confidence interval of:

[0.1677, 0.2268]

These results provide model-localization and deletion-based faithfulness evidence. They do not establish causal reasoning, clinical interpretability, or equivalence to expert radiological decision-making.

Independent External Validation

The frozen model was evaluated on BrEaST and BUS-UCLM without:

retraining;
fine-tuning;
calibration;
threshold optimization;
test-time augmentation; or
performance-based case filtering.
Diagnostic Performance
Metric	BrEaST	BUS-UCLM
Three-class balanced accuracy	0.4958	0.5407
Macro-F1	0.3187	0.5371
Macro ROC-AUC	0.6898	0.7098
Cancer sensitivity	0.2245	0.2556
Cancer specificity	0.9304	0.9444
Cancer F1-score	0.3359	0.3151
Cancer ROC-AUC	0.6573	0.6517
Average precision	0.5299	0.2762

BrEaST three-class results should be interpreted descriptively because only four normal cases are available.

External Lesion Segmentation
Metric	BrEaST lesion subset	BUS-UCLM lesion subset
Dice	0.6148	0.5711
IoU	0.5033	0.4698
Precision	0.7065	0.6166
Recall	0.6207	0.6386

Internal BUSI and external segmentation values should not be treated as directly matched cross-dataset differences because the evaluation populations differ.

Reproducibility Environment

The frozen development environment was:

Component	Version / Configuration
Python	3.10.20
PyTorch	2.5.1+cu121
torchvision	0.20.1+cu121
NumPy	2.2.6
CUDA	12.1
cuDNN	90100
GPU	NVIDIA GeForce RTX 3060, 12 GB
System RAM	32 GB
Random seed	20260830

Deterministic PyTorch execution was enabled during the frozen experiments.

Reproducibility Notes

To reproduce the reported experiments:

obtain BUSI, BrEaST, and BUS-UCLM from their official repositories;
preserve the dataset-specific licensing conditions;
configure dataset locations without modifying the frozen preprocessing procedure;
use the provided configuration and split information;
reproduce development experiments using seed 20260830;
keep the final BUSI holdout isolated from model development;
do not optimize thresholds or model settings using external datasets.

The final holdout and external datasets must not be used for architecture selection, hyperparameter tuning, or post hoc calibration if the reported protocol is to be reproduced faithfully.

