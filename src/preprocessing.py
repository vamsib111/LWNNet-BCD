"""
Public preprocessing interface for LWNNet-BCD.

The executed and authenticated preprocessing implementation is preserved in
``src.dataset``.  This module only re-exports that implementation through a
stable public interface and does not duplicate or modify preprocessing logic.
"""

from .dataset import (
    GAUSSIAN_KERNEL,
    GAUSSIAN_SIGMA,
    HORIZONTAL_FLIP_PROBABILITY,
    IMAGE_SIZE,
    MEDIAN_KERNEL,
    ROTATION_LIMIT_DEGREES,
    ROTATION_MAX_DEGREES,
    ROTATION_MIN_DEGREES,
    minmax_normalize,
    preprocess_image,
    synchronized_train_augmentation,
    apply_exact_r4_preprocessing,
)

# Manuscript-facing neutral alias.
apply_lwnnet_bcd_preprocessing = apply_exact_r4_preprocessing

__all__ = [
    "GAUSSIAN_KERNEL",
    "GAUSSIAN_SIGMA",
    "HORIZONTAL_FLIP_PROBABILITY",
    "IMAGE_SIZE",
    "MEDIAN_KERNEL",
    "ROTATION_LIMIT_DEGREES",
    "ROTATION_MAX_DEGREES",
    "ROTATION_MIN_DEGREES",
    "minmax_normalize",
    "preprocess_image",
    "synchronized_train_augmentation",
    "apply_exact_r4_preprocessing",
    "apply_lwnnet_bcd_preprocessing",
]
