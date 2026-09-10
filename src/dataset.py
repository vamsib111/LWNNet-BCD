# ==================================================================================================
# LWNNet-BCD
# FROZEN R4 DEVELOPMENT DATA PIPELINE
#
# Exact historical preprocessing / augmentation functions recovered from execution history.
#
# Resize ownership was explicitly reconciled by R4-03R2:
#
#   * historical preprocess_image performs resize -> use it directly
#   * otherwise -> apply controlled 128x128 INTER_AREA resize immediately before preprocess_image
#
# ==================================================================================================

from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

GAUSSIAN_KERNEL = (3, 3)
GAUSSIAN_SIGMA = 0.0
HORIZONTAL_FLIP_PROBABILITY = 0.5
IMAGE_SIZE = 128
MEDIAN_KERNEL = 3
ROTATION_LIMIT_DEGREES = 15.0
ROTATION_MAX_DEGREES = 15.0
ROTATION_MIN_DEGREES = -15.0
height = 224
width = 224

def minmax_normalize(image):
    """
    Per-image min-max normalization to [0, 1].
    """

    image = image.astype(np.float32)

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32
        )

    return image.astype(np.float32)

def preprocess_image(
    image_rgb,
    variant="full"
):
    """
    Preprocesses the resized RGB ultrasound image.

    This function does NOT perform augmentation.
    """

    image = image_rgb.copy()

    # ----------------------------------------------------------------------------------------------
    # Median filtering
    # ----------------------------------------------------------------------------------------------

    if variant in {
        "full",
        "no_gaussian",
        "no_minmax",
    }:

        image = cv2.medianBlur(
            image,
            MEDIAN_KERNEL
        )

    # ----------------------------------------------------------------------------------------------
    # Gaussian filtering
    # ----------------------------------------------------------------------------------------------

    if variant in {
        "full",
        "no_median",
        "no_minmax",
    }:

        image = cv2.GaussianBlur(
            image,
            GAUSSIAN_KERNEL,
            GAUSSIAN_SIGMA
        )

    # ----------------------------------------------------------------------------------------------
    # Intensity normalization
    # ----------------------------------------------------------------------------------------------

    if variant in {
        "full",
        "no_median",
        "no_gaussian",
        "no_filters",
    }:

        image = minmax_normalize(
            image
        )

    else:

        # Fixed numerical scaling only.
        # This is intentionally distinct from adaptive per-image min-max normalization.
        image = (
            image.astype(np.float32)
            / 255.0
        )

    return image.astype(np.float32)

def synchronized_train_augmentation(
    image,
    mask
):
    """
    Applies identical spatial transformations to image and mask.

    Image:
        bilinear interpolation for rotation.

    Mask:
        nearest-neighbour interpolation only.
    """

    # ----------------------------------------------------------------------------------------------
    # Random horizontal flip
    # ----------------------------------------------------------------------------------------------

    if torch.rand(1).item() < HORIZONTAL_FLIP_PROBABILITY:

        image = cv2.flip(
            image,
            1
        )

        mask = cv2.flip(
            mask,
            1
        )

    # ----------------------------------------------------------------------------------------------
    # Random rotation in [-15, +15] degrees
    # ----------------------------------------------------------------------------------------------

    angle = torch.empty(
        1
    ).uniform_(
        -ROTATION_LIMIT_DEGREES,
        ROTATION_LIMIT_DEGREES
    ).item()

    height, width = mask.shape

    rotation_matrix = cv2.getRotationMatrix2D(
        center=(
            width / 2.0,
            height / 2.0
        ),
        angle=angle,
        scale=1.0
    )

    image = cv2.warpAffine(
        image,
        rotation_matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return image, mask


PREPROCESS_CONTAINS_RESIZE = False
RESIZE_EXECUTION_OWNER = 'R4_DATASET_WRAPPER_BEFORE_PREPROCESS'
IMAGE_DECODE_MODE = 'COLOR'
CONVERT_BGR_TO_RGB = True

CLASS_TO_INDEX = {
    "normal": 0,
    "benign": 1,
    "malignant": 2,
}


def apply_exact_r4_preprocessing(
    raw_image
):

    if PREPROCESS_CONTAINS_RESIZE:

        processed = preprocess_image(
            raw_image
        )


    else:

        resized = cv2.resize(
            raw_image,
            (
                int(
                    IMAGE_SIZE
                ),
                int(
                    IMAGE_SIZE
                ),
            ),
            interpolation=cv2.INTER_AREA,
        )


        processed = preprocess_image(
            resized
        )


    processed = np.asarray(
        processed,
        dtype=np.float32,
    )


    if processed.shape[0] != int(
        IMAGE_SIZE
    ):

        raise RuntimeError(
            f"Preprocessed image height mismatch: {processed.shape}"
        )


    if processed.shape[1] != int(
        IMAGE_SIZE
    ):

        raise RuntimeError(
            f"Preprocessed image width mismatch: {processed.shape}"
        )


    if not np.isfinite(
        processed
    ).all():

        raise RuntimeError(
            "Preprocessed image contains non-finite values."
        )


    if float(
        processed.min()
    ) < -1e-6:

        raise RuntimeError(
            "Preprocessed image minimum is below zero."
        )


    if float(
        processed.max()
    ) > 1.0 + 1e-6:

        raise RuntimeError(
            "Preprocessed image maximum exceeds one."
        )


    return processed


class R4DevelopmentDataset(
    Dataset
):

    def __init__(
        self,
        records,
        augment=False,
    ):

        self.records = list(
            records
        )

        self.augment = bool(
            augment
        )


    def __len__(
        self
    ):

        return len(
            self.records
        )


    @staticmethod
    def _load_union_mask(
        mask_paths
    ):

        union_mask = None

        native_shape = None


        for mask_path in mask_paths:

            mask = cv2.imread(
                str(
                    mask_path
                ),
                cv2.IMREAD_GRAYSCALE,
            )


            if mask is None:

                raise RuntimeError(
                    f"Failed to read development mask: {mask_path}"
                )


            binary = (
                mask
                >
                127
            ).astype(
                np.uint8
            )


            if union_mask is None:

                union_mask = binary.copy()

                native_shape = tuple(
                    binary.shape
                )


            else:

                if tuple(
                    binary.shape
                ) != native_shape:

                    raise RuntimeError(
                        "Multi-mask native shape mismatch."
                    )


                union_mask = np.logical_or(
                    union_mask,
                    binary,
                ).astype(
                    np.uint8
                )


        if union_mask is None:

            raise RuntimeError(
                "No development mask loaded."
            )


        return union_mask


    def __getitem__(
        self,
        index
    ):

        record = self.records[
            index
        ]


        image_path = Path(
            record[
                "image_path"
            ]
        )


        mask_paths = [
            Path(
                path
            )

            for path in record[
                "mask_paths"
            ]
        ]


        if IMAGE_DECODE_MODE == "GRAYSCALE":

            image = cv2.imread(
                str(
                    image_path
                ),
                cv2.IMREAD_GRAYSCALE,
            )


        else:

            image = cv2.imread(
                str(
                    image_path
                ),
                cv2.IMREAD_COLOR,
            )


        if image is None:

            raise RuntimeError(
                f"Failed to read development ultrasound image: {image_path}"
            )


        if (
            image.ndim == 3
            and
            CONVERT_BGR_TO_RGB
        ):

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )


        mask = self._load_union_mask(
            mask_paths
        )


        image = apply_exact_r4_preprocessing(
            image
        )


        mask = cv2.resize(
            mask,
            (
                int(
                    IMAGE_SIZE
                ),
                int(
                    IMAGE_SIZE
                ),
            ),
            interpolation=cv2.INTER_NEAREST,
        )


        mask = (
            mask
            >
            0
        ).astype(
            np.float32
        )


        if self.augment:

            image, mask = synchronized_train_augmentation(
                image,
                mask,
            )


        image = np.asarray(
            image,
            dtype=np.float32,
        )


        mask = (
            np.asarray(
                mask
            )
            >=
            0.5
        ).astype(
            np.float32
        )


        if image.ndim == 2:

            image_tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        image
                    )
                )
                .unsqueeze(
                    0
                )
                .repeat(
                    3,
                    1,
                    1,
                )
                .float()
            )


        elif (
            image.ndim == 3
            and
            image.shape[
                2
            ] == 3
        ):

            image_tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(
                        np.transpose(
                            image,
                            (
                                2,
                                0,
                                1,
                            )
                        )
                    )
                )
                .float()
            )


        else:

            raise RuntimeError(
                f"Unsupported processed image shape: {image.shape}"
            )


        mask_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    mask
                )
            )
            .unsqueeze(
                0
            )
            .float()
        )


        class_name = str(
            record[
                "class"
            ]
        ).strip().lower()


        diagnostic_target = torch.tensor(
            CLASS_TO_INDEX[
                class_name
            ],
            dtype=torch.long,
        )


        return {
            "image":
                image_tensor,

            "mask":
                mask_tensor,

            "diagnostic_target":
                diagnostic_target,

            "filename":
                record[
                    "filename"
                ],

            "class_name":
                class_name,
        }
