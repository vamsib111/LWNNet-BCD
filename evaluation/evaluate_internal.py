"""
Reusable internal evaluation driver for LWNNet-BCD.

This module reproduces the authenticated internal evaluation semantics without
performing model execution or dataset/file discovery.

Protocol:
    * class order is Normal, Benign, Malignant = 0, 1, 2;
    * three-class and malignant-vs-rest metrics use all supplied cases;
    * cancer score is P(Malignant), with threshold 0.5;
    * segmentation probabilities are thresholded at 0.5;
    * Dice, IoU, pixel precision, and pixel recall are summarized only for
      lesion cases, defined as class index 1 or 2;
    * Normal cases are summarized separately using predicted foreground
      fraction;
    * lesion metrics are case-level values followed by an arithmetic mean;
    * benign and malignant Dice are retained separately;
    * optional boundary logits are transformed by sigmoid and thresholded at
      0.5, then summarized over lesion cases only.

Historical internal-evaluation authority SHA256:
    1f2340eedfa4376ff6ae23e6dfc426eb2f56e10058fc62b88df697fb42f4f9fc
"""

import numpy as np
import torch

from .metrics import (
    cancer_metrics,
    multiclass_metrics,
    spatial_metrics_per_sample,
)


CLASS_ORDER = (
    "Normal",
    "Benign",
    "Malignant",
)

NORMAL_CLASS_INDEX = 0
BENIGN_CLASS_INDEX = 1
MALIGNANT_CLASS_INDEX = 2

SEGMENTATION_THRESHOLD = 0.5
BOUNDARY_THRESHOLD = 0.5
CANCER_THRESHOLD = 0.5


def _as_targets(
    targets,
):
    targets = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(
        -1
    )

    if targets.size == 0:
        raise ValueError(
            "targets must contain at least one case."
        )

    if not set(
        np.unique(
            targets
        ).tolist()
    ).issubset(
        {
            NORMAL_CLASS_INDEX,
            BENIGN_CLASS_INDEX,
            MALIGNANT_CLASS_INDEX,
        }
    ):
        raise ValueError(
            "targets must use class indices 0, 1, and 2."
        )

    return targets


def _as_probabilities(
    probabilities,
    n_cases,
):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.shape != (
        n_cases,
        3,
    ):
        raise ValueError(
            "diagnostic probabilities must have shape (N, 3)."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "diagnostic probabilities must be finite."
        )

    if np.any(
        probabilities
        <
        0.0
    ) or np.any(
        probabilities
        >
        1.0
    ):
        raise ValueError(
            "diagnostic probabilities must lie in [0, 1]."
        )

    probability_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            "each diagnostic probability row must sum to 1."
        )

    return probabilities


def _as_spatial_tensor(
    values,
    name,
):
    if torch.is_tensor(
        values
    ):
        tensor = (
            values
            .detach()
            .cpu()
            .float()
        )

    else:
        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(
            1
        )

    if (
        tensor.ndim
        !=
        4
        or
        tensor.shape[
            1
        ]
        !=
        1
    ):
        raise ValueError(
            f"{name} must have shape (N,H,W) or (N,1,H,W)."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return tensor


def _mean_metric(
    values,
    indices,
):
    return float(
        values[
            indices
        ].mean().item()
    )


def evaluate_internal(
    targets,
    diagnostic_probabilities,
    segmentation_probabilities,
    segmentation_targets,
    segmentation_threshold=SEGMENTATION_THRESHOLD,
    boundary_logits=None,
    boundary_targets=None,
    boundary_threshold=BOUNDARY_THRESHOLD,
):
    """
    Evaluate already available predictions using the authenticated internal
    LWNNet-BCD evaluation protocol.

    No model execution or file loading is performed.
    """

    targets = _as_targets(
        targets
    )

    n_cases = int(
        targets.size
    )

    diagnostic_probabilities = _as_probabilities(
        diagnostic_probabilities,
        n_cases,
    )

    segmentation_probabilities = _as_spatial_tensor(
        segmentation_probabilities,
        "segmentation_probabilities",
    )

    segmentation_targets = _as_spatial_tensor(
        segmentation_targets,
        "segmentation_targets",
    )

    if segmentation_probabilities.shape != segmentation_targets.shape:
        raise ValueError(
            "segmentation prediction/target shapes must match."
        )

    if segmentation_probabilities.shape[
        0
    ] != n_cases:
        raise ValueError(
            "segmentation case count must match targets."
        )

    if np.any(
        segmentation_threshold
        !=
        SEGMENTATION_THRESHOLD
    ):
        raise ValueError(
            "authenticated internal segmentation threshold is fixed at 0.5."
        )

    if np.any(
        boundary_threshold
        !=
        BOUNDARY_THRESHOLD
    ):
        raise ValueError(
            "authenticated internal boundary threshold is fixed at 0.5."
        )

    segmentation_prediction = (
        segmentation_probabilities
        >=
        float(
            segmentation_threshold
        )
    ).float()

    segmentation_metrics = spatial_metrics_per_sample(
        segmentation_prediction,
        segmentation_targets,
    )

    lesion_indices_np = np.flatnonzero(
        targets
        !=
        NORMAL_CLASS_INDEX
    )

    normal_indices_np = np.flatnonzero(
        targets
        ==
        NORMAL_CLASS_INDEX
    )

    benign_indices_np = np.flatnonzero(
        targets
        ==
        BENIGN_CLASS_INDEX
    )

    malignant_indices_np = np.flatnonzero(
        targets
        ==
        MALIGNANT_CLASS_INDEX
    )

    if lesion_indices_np.size == 0:
        raise ValueError(
            "at least one lesion case is required."
        )

    if normal_indices_np.size == 0:
        raise ValueError(
            "at least one Normal case is required."
        )

    if benign_indices_np.size == 0:
        raise ValueError(
            "at least one Benign case is required."
        )

    if malignant_indices_np.size == 0:
        raise ValueError(
            "at least one Malignant case is required."
        )

    lesion_indices = torch.as_tensor(
        lesion_indices_np,
        dtype=torch.long,
    )

    normal_indices = torch.as_tensor(
        normal_indices_np,
        dtype=torch.long,
    )

    benign_indices = torch.as_tensor(
        benign_indices_np,
        dtype=torch.long,
    )

    malignant_indices = torch.as_tensor(
        malignant_indices_np,
        dtype=torch.long,
    )

    segmentation_summary = {
        "dice":
            _mean_metric(
                segmentation_metrics[
                    "dice"
                ],
                lesion_indices,
            ),

        "iou":
            _mean_metric(
                segmentation_metrics[
                    "iou"
                ],
                lesion_indices,
            ),

        "precision":
            _mean_metric(
                segmentation_metrics[
                    "precision"
                ],
                lesion_indices,
            ),

        "recall":
            _mean_metric(
                segmentation_metrics[
                    "recall"
                ],
                lesion_indices,
            ),

        "benign_dice":
            _mean_metric(
                segmentation_metrics[
                    "dice"
                ],
                benign_indices,
            ),

        "malignant_dice":
            _mean_metric(
                segmentation_metrics[
                    "dice"
                ],
                malignant_indices,
            ),
    }

    normal_localization_summary = {
        "normal_false_positive_fraction":
            _mean_metric(
                segmentation_metrics[
                    "fp_fraction"
                ],
                normal_indices,
            ),
    }

    diagnostic_summary = multiclass_metrics(
        targets,
        diagnostic_probabilities,
    )

    cancer_summary = cancer_metrics(
        targets,
        diagnostic_probabilities[
            :,
            MALIGNANT_CLASS_INDEX,
        ],
    )

    result = {
        "class_order":
            CLASS_ORDER,

        "thresholds":
            {
                "segmentation":
                    float(
                        segmentation_threshold
                    ),

                "cancer":
                    CANCER_THRESHOLD,

                "boundary":
                    (
                        float(
                            boundary_threshold
                        )
                        if boundary_logits is not None
                        else None
                    ),
            },

        "counts":
            {
                "all":
                    n_cases,

                "normal":
                    int(
                        normal_indices_np.size
                    ),

                "benign":
                    int(
                        benign_indices_np.size
                    ),

                "malignant":
                    int(
                        malignant_indices_np.size
                    ),

                "lesion":
                    int(
                        lesion_indices_np.size
                    ),
            },

        "segmentation_lesion_only":
            segmentation_summary,

        "normal_localization":
            normal_localization_summary,

        "three_class_diagnosis":
            diagnostic_summary,

        "malignant_vs_rest_cancer_detection":
            cancer_summary,
    }

    if (
        boundary_logits is None
        !=
        (
            boundary_targets is None
        )
    ):
        raise ValueError(
            "boundary_logits and boundary_targets must be supplied together."
        )

    if boundary_logits is not None:

        boundary_logits = _as_spatial_tensor(
            boundary_logits,
            "boundary_logits",
        )

        boundary_targets = _as_spatial_tensor(
            boundary_targets,
            "boundary_targets",
        )

        if boundary_logits.shape != boundary_targets.shape:
            raise ValueError(
                "boundary logit/target shapes must match."
            )

        if boundary_logits.shape[
            0
        ] != n_cases:
            raise ValueError(
                "boundary case count must match targets."
            )

        boundary_prediction = (
            torch.sigmoid(
                boundary_logits
            )
            >=
            float(
                boundary_threshold
            )
        ).float()

        boundary_metrics = spatial_metrics_per_sample(
            boundary_prediction,
            boundary_targets,
        )

        result[
            "boundary_lesion_only"
        ] = {
            "dice":
                _mean_metric(
                    boundary_metrics[
                        "dice"
                    ],
                    lesion_indices,
                ),

            "iou":
                _mean_metric(
                    boundary_metrics[
                        "iou"
                    ],
                    lesion_indices,
                ),

            "precision":
                _mean_metric(
                    boundary_metrics[
                        "precision"
                    ],
                    lesion_indices,
                ),

            "recall":
                _mean_metric(
                    boundary_metrics[
                        "recall"
                    ],
                    lesion_indices,
                ),
        }

    return result


__all__ = [
    "CLASS_ORDER",
    "SEGMENTATION_THRESHOLD",
    "BOUNDARY_THRESHOLD",
    "CANCER_THRESHOLD",
    "evaluate_internal",
]
