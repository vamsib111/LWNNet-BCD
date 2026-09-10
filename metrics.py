"""
LWNNet-BCD reusable evaluation metrics.

The three metric functions in this module are exact function-level exports
from the authenticated historical final-holdout evaluation implementation.

Historical metric authority SHA256:
    1f2340eedfa4376ff6ae23e6dfc426eb2f56e10058fc62b88df697fb42f4f9fc

Important spatial-metric note:
    spatial_metrics_per_sample() computes per-sample binary spatial metrics.
    It does not itself select a lesion-only cohort. Any cohort/sample selection
    required by an evaluation protocol must be performed by the evaluation
    driver before aggregate reporting.

Cancer sensitivity is computed directly from the binary confusion matrix as
TP / (TP + FN), and specificity as TN / (TN + FP), matching the authenticated
historical implementation.
"""

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CANCER_THRESHOLD = 0.5

def spatial_metrics_per_sample(
    prediction,
    target,
):

    prediction = prediction.float()

    target = target.float()


    prediction_flat = prediction.flatten(
        1
    )

    target_flat = target.flatten(
        1
    )


    intersection = (
        prediction_flat
        *
        target_flat
    ).sum(
        dim=1
    )


    predicted_positive = prediction_flat.sum(
        dim=1
    )


    actual_positive = target_flat.sum(
        dim=1
    )


    union = (
        predicted_positive
        +
        actual_positive
        -
        intersection
    )


    dice = (
        2.0
        *
        intersection
        +
        1e-7
    ) / (
        predicted_positive
        +
        actual_positive
        +
        1e-7
    )


    iou = (
        intersection
        +
        1e-7
    ) / (
        union
        +
        1e-7
    )


    precision = torch.where(
        predicted_positive
        >
        0,
        intersection
        /
        predicted_positive.clamp_min(
            1e-7
        ),
        torch.zeros_like(
            intersection
        ),
    )


    recall = torch.where(
        actual_positive
        >
        0,
        intersection
        /
        actual_positive.clamp_min(
            1e-7
        ),
        torch.zeros_like(
            intersection
        ),
    )


    fp_fraction = prediction_flat.mean(
        dim=1
    )


    return {
        "dice":
            dice,

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall,

        "fp_fraction":
            fp_fraction,
    }


def multiclass_metrics(
    targets,
    probabilities,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    predictions = np.argmax(
        probabilities,
        axis=1,
    )


    recalls = recall_score(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )


    class_aucs = {}


    for target_value, class_name in [
        (0, "normal"),
        (1, "benign"),
        (2, "malignant"),
    ]:

        binary_target = (
            targets
            ==
            target_value
        ).astype(
            np.int64
        )


        class_aucs[
            class_name
        ] = float(
            roc_auc_score(
                binary_target,
                probabilities[
                    :,
                    target_value
                ],
            )
        )


    return {
        "accuracy":
            float(
                accuracy_score(
                    targets,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    targets,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    targets,
                    predictions,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_auc_ovr":
            float(
                roc_auc_score(
                    targets,
                    probabilities,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                    multi_class="ovr",
                    average="macro",
                )
            ),

        "normal_recall":
            float(
                recalls[
                    0
                ]
            ),

        "benign_recall":
            float(
                recalls[
                    1
                ]
            ),

        "malignant_recall":
            float(
                recalls[
                    2
                ]
            ),

        "normal_auc_ovr":
            class_aucs[
                "normal"
            ],

        "benign_auc_ovr":
            class_aucs[
                "benign"
            ],

        "malignant_auc_ovr":
            class_aucs[
                "malignant"
            ],

        "confusion_matrix":
            confusion_matrix(
                targets,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
            ).tolist(),
    }


def cancer_metrics(
    targets,
    malignant_probability,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )


    malignant_probability = np.asarray(
        malignant_probability,
        dtype=np.float64,
    )


    cancer_target = (
        targets
        ==
        2
    ).astype(
        np.int64
    )


    cancer_prediction = (
        malignant_probability
        >=
        CANCER_THRESHOLD
    ).astype(
        np.int64
    )


    matrix = confusion_matrix(
        cancer_target,
        cancer_prediction,
        labels=[
            0,
            1,
        ],
    )


    tn, fp, fn, tp = matrix.ravel()


    sensitivity = float(
        tp
        /
        (
            tp
            +
            fn
        )
    )


    specificity = float(
        tn
        /
        (
            tn
            +
            fp
        )
    )


    return {
        "accuracy_at_0_5":
            float(
                accuracy_score(
                    cancer_target,
                    cancer_prediction,
                )
            ),

        "precision_at_0_5":
            float(
                precision_score(
                    cancer_target,
                    cancer_prediction,
                    zero_division=0,
                )
            ),

        "sensitivity_at_0_5":
            sensitivity,

        "specificity_at_0_5":
            specificity,

        "f1_at_0_5":
            float(
                f1_score(
                    cancer_target,
                    cancer_prediction,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    cancer_target,
                    malignant_probability,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    cancer_target,
                    malignant_probability,
                )
            ),

        "confusion_matrix":
            matrix.tolist(),
    }


__all__ = [
    "spatial_metrics_per_sample",
    "multiclass_metrics",
    "cancer_metrics",
]
