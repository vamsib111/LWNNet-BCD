"""
External evaluation utilities for LWNNet-BCD.

This module is a controlled public adaptation of the authenticated external
diagnostic and segmentation evaluation protocols. It operates on already
available predictions and reference arrays only.

Diagnostic protocol:
    * class order is Normal, Benign, Malignant = 0, 1, 2;
    * three-class prediction is argmax over native probabilities;
    * malignant-versus-rest score is the Malignant probability;
    * cancer threshold is fixed at 0.5;
    * mathematically undefined class-dependent metrics are returned as NaN.

Lesion-segmentation protocol:
    * threshold is fixed at 0.5;
    * only lesion-present targets are valid;
    * case metrics use unsmoothed TP/FP/FN arithmetic;
    * Dice = 2TP / (2TP + FP + FN);
    * IoU = TP / (TP + FP + FN);
    * pixel precision = TP / (TP + FP), or 0 for an empty prediction;
    * pixel recall = TP / (TP + FN);
    * dataset point estimates are arithmetic means of case-level metrics.

Normal false-positive protocol:
    * foreground fraction;
    * any-positive case indicator;
    * mean segmentation probability;
    * empty-target Dice and IoU are intentionally not reported.

Bootstrap confidence intervals are intentionally implemented separately.

Historical external diagnostic authority SHA256:
    0b9af4680512306b7da919876b2293d8d8e436632804876c9f6b401019bcc531

Historical external segmentation authority SHA256:
    fd5dc664c0bb6cd237c458dcafef9a10e62ca5bf8e1ce13762a8676a24ce4d12
"""

import math

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


CLASS_ORDER = (
    "Normal",
    "Benign",
    "Malignant",
)

NORMAL_INDEX = 0
BENIGN_INDEX = 1
MALIGNANT_INDEX = 2

CANCER_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = 0.5


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return float(
            "nan"
        )

    return float(
        numerator
        /
        denominator
    )


def _validate_three_class_inputs(
    y_true,
    probabilities,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(
        -1
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    if y_true.size == 0:
        raise ValueError(
            "at least one target is required."
        )


    if probabilities.shape != (
        y_true.size,
        3,
    ):
        raise ValueError(
            "probabilities must have shape (N, 3)."
        )


    if not set(
        np.unique(
            y_true
        ).tolist()
    ).issubset(
        {
            NORMAL_INDEX,
            BENIGN_INDEX,
            MALIGNANT_INDEX,
        }
    ):
        raise ValueError(
            "targets must use class indices 0, 1, and 2."
        )


    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "probabilities must be finite."
        )


    if (
        np.any(
            probabilities
            <
            0.0
        )
        or
        np.any(
            probabilities
            >
            1.0
        )
    ):
        raise ValueError(
            "probabilities must lie in [0, 1]."
        )


    if not np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            "each probability row must sum to 1."
        )


    return (
        y_true,
        probabilities,
    )


def three_class_metrics(
    y_true,
    probabilities,
):
    y_true, probabilities = _validate_three_class_inputs(
        y_true,
        probabilities,
    )


    y_pred = np.argmax(
        probabilities,
        axis=1,
    )


    confusion = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.int64,
    )


    for true_label, predicted_label in zip(
        y_true,
        y_pred,
    ):
        confusion[
            int(
                true_label
            ),
            int(
                predicted_label
            ),
        ] += 1


    accuracy = float(
        np.mean(
            y_pred
            ==
            y_true
        )
    )


    recalls = []

    f1_values = []

    auc_values = []


    for class_index in range(
        3
    ):

        tp = int(
            confusion[
                class_index,
                class_index,
            ]
        )

        fn = int(
            confusion[
                class_index,
                :
            ].sum()
            -
            tp
        )

        fp = int(
            confusion[
                :,
                class_index
            ].sum()
            -
            tp
        )


        recall = safe_divide(
            tp,
            tp
            +
            fn,
        )


        precision = safe_divide(
            tp,
            tp
            +
            fp,
        )


        if (
            math.isfinite(
                recall
            )
            and
            math.isfinite(
                precision
            )
            and
            (
                recall
                +
                precision
            )
            >
            0.0
        ):

            f1 = float(
                2.0
                *
                recall
                *
                precision
                /
                (
                    recall
                    +
                    precision
                )
            )


        elif (
            math.isfinite(
                recall
            )
            and
            math.isfinite(
                precision
            )
        ):

            f1 = 0.0


        else:

            f1 = float(
                "nan"
            )


        recalls.append(
            recall
        )

        f1_values.append(
            f1
        )


        binary_true = (
            y_true
            ==
            class_index
        ).astype(
            np.int64
        )


        positives = int(
            binary_true.sum()
        )

        negatives = int(
            len(
                binary_true
            )
            -
            positives
        )


        if (
            positives
            >
            0
            and
            negatives
            >
            0
        ):

            auc_value = float(
                roc_auc_score(
                    binary_true,
                    probabilities[
                        :,
                        class_index,
                    ],
                )
            )


        else:

            auc_value = float(
                "nan"
            )


        auc_values.append(
            auc_value
        )


    balanced_accuracy = (
        float(
            np.mean(
                recalls
            )
        )
        if all(
            math.isfinite(
                value
            )
            for value in recalls
        )
        else float(
            "nan"
        )
    )


    macro_f1 = (
        float(
            np.mean(
                f1_values
            )
        )
        if all(
            math.isfinite(
                value
            )
            for value in f1_values
        )
        else float(
            "nan"
        )
    )


    macro_auc = (
        float(
            np.mean(
                auc_values
            )
        )
        if all(
            math.isfinite(
                value
            )
            for value in auc_values
        )
        else float(
            "nan"
        )
    )


    metrics = {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "recall_normal":
            recalls[
                0
            ],

        "recall_benign":
            recalls[
                1
            ],

        "recall_malignant":
            recalls[
                2
            ],

        "auc_normal_ovr":
            auc_values[
                0
            ],

        "auc_benign_ovr":
            auc_values[
                1
            ],

        "auc_malignant_ovr":
            auc_values[
                2
            ],

        "macro_auc_ovr":
            macro_auc,
    }


    return (
        metrics,
        confusion,
    )


def cancer_metrics(
    y_true_three_class,
    probabilities,
    threshold=CANCER_THRESHOLD,
):
    if float(
        threshold
    ) != CANCER_THRESHOLD:
        raise ValueError(
            "cancer threshold is fixed at 0.5."
        )


    y_true_three_class, probabilities = _validate_three_class_inputs(
        y_true_three_class,
        probabilities,
    )


    scores = probabilities[
        :,
        MALIGNANT_INDEX,
    ]


    y_true = (
        y_true_three_class
        ==
        MALIGNANT_INDEX
    ).astype(
        np.int64
    )


    y_pred = (
        scores
        >=
        float(
            threshold
        )
    ).astype(
        np.int64
    )


    tp = int(
        np.sum(
            (
                y_true
                ==
                1
            )
            &
            (
                y_pred
                ==
                1
            )
        )
    )


    tn = int(
        np.sum(
            (
                y_true
                ==
                0
            )
            &
            (
                y_pred
                ==
                0
            )
        )
    )


    fp = int(
        np.sum(
            (
                y_true
                ==
                0
            )
            &
            (
                y_pred
                ==
                1
            )
        )
    )


    fn = int(
        np.sum(
            (
                y_true
                ==
                1
            )
            &
            (
                y_pred
                ==
                0
            )
        )
    )


    accuracy = safe_divide(
        tp
        +
        tn,
        tp
        +
        tn
        +
        fp
        +
        fn,
    )


    precision = safe_divide(
        tp,
        tp
        +
        fp,
    )


    sensitivity = safe_divide(
        tp,
        tp
        +
        fn,
    )


    specificity = safe_divide(
        tn,
        tn
        +
        fp,
    )


    f1 = safe_divide(
        2
        *
        tp,
        2
        *
        tp
        +
        fp
        +
        fn,
    )


    positives = int(
        y_true.sum()
    )

    negatives = int(
        len(
            y_true
        )
        -
        positives
    )


    if (
        positives
        >
        0
        and
        negatives
        >
        0
    ):

        roc_auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )


        average_precision = float(
            average_precision_score(
                y_true,
                scores,
            )
        )


    else:

        roc_auc = float(
            "nan"
        )

        average_precision = float(
            "nan"
        )


    metrics = {
        "accuracy":
            accuracy,

        "precision":
            precision,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

        "average_precision":
            average_precision,
    }


    confusion = {
        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,
    }


    return (
        metrics,
        confusion,
    )


def lesion_segmentation_case_metrics(
    segmentation_probability,
    target,
    threshold=SEGMENTATION_THRESHOLD,
):
    if float(
        threshold
    ) != SEGMENTATION_THRESHOLD:
        raise ValueError(
            "segmentation threshold is fixed at 0.5."
        )


    probability = np.asarray(
        segmentation_probability,
        dtype=np.float64,
    )


    target_array = np.asarray(
        target
    )


    if probability.shape != target_array.shape:
        raise ValueError(
            "segmentation probability and target shapes must match."
        )


    if probability.size == 0:
        raise ValueError(
            "segmentation arrays must be non-empty."
        )


    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "segmentation probabilities must be finite."
        )


    if (
        np.any(
            probability
            <
            0.0
        )
        or
        np.any(
            probability
            >
            1.0
        )
    ):
        raise ValueError(
            "segmentation probabilities must lie in [0, 1]."
        )


    if not set(
        np.unique(
            target_array
        ).tolist()
    ).issubset(
        {
            0,
            1,
            False,
            True,
        }
    ):
        raise ValueError(
            "segmentation targets must be binary."
        )


    target_binary = target_array.astype(
        bool
    )


    if int(
        target_binary.sum()
    ) <= 0:
        raise ValueError(
            "lesion segmentation requires a non-empty target."
        )


    prediction_binary = (
        probability
        >=
        float(
            threshold
        )
    )


    tp = int(
        np.logical_and(
            prediction_binary,
            target_binary,
        ).sum()
    )


    fp = int(
        np.logical_and(
            prediction_binary,
            np.logical_not(
                target_binary
            ),
        ).sum()
    )


    fn = int(
        np.logical_and(
            np.logical_not(
                prediction_binary
            ),
            target_binary,
        ).sum()
    )


    true_pixels = int(
        target_binary.sum()
    )


    predicted_pixels = int(
        prediction_binary.sum()
    )


    dice_denominator = (
        2
        *
        tp
        +
        fp
        +
        fn
    )


    iou_denominator = (
        tp
        +
        fp
        +
        fn
    )


    recall_denominator = (
        tp
        +
        fn
    )


    dice = float(
        (
            2
            *
            tp
        )
        /
        dice_denominator
    )


    iou = float(
        tp
        /
        iou_denominator
    )


    pixel_precision = (
        float(
            tp
            /
            (
                tp
                +
                fp
            )
        )
        if (
            tp
            +
            fp
        )
        >
        0
        else
        0.0
    )


    pixel_recall = float(
        tp
        /
        recall_denominator
    )


    return {
        "true_foreground_pixels":
            true_pixels,

        "predicted_foreground_pixels":
            predicted_pixels,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "dice":
            dice,

        "iou":
            iou,

        "pixel_precision":
            pixel_precision,

        "pixel_recall":
            pixel_recall,
    }


def evaluate_lesion_segmentation(
    segmentation_probabilities,
    segmentation_targets,
    threshold=SEGMENTATION_THRESHOLD,
):
    probabilities = np.asarray(
        segmentation_probabilities,
        dtype=np.float64,
    )


    targets = np.asarray(
        segmentation_targets
    )


    if probabilities.shape != targets.shape:
        raise ValueError(
            "segmentation probability and target arrays must have identical shapes."
        )


    if probabilities.ndim < 2:
        raise ValueError(
            "segmentation arrays must have shape (N, ...)."
        )


    if probabilities.shape[
        0
    ] == 0:
        raise ValueError(
            "at least one lesion case is required."
        )


    case_rows = []


    for case_index in range(
        probabilities.shape[
            0
        ]
    ):

        case_rows.append(
            lesion_segmentation_case_metrics(
                probabilities[
                    case_index
                ],
                targets[
                    case_index
                ],
                threshold=threshold,
            )
        )


    metric_names = (
        "dice",
        "iou",
        "pixel_precision",
        "pixel_recall",
    )


    summary = {
        metric_name:
            float(
                np.mean(
                    [
                        row[
                            metric_name
                        ]

                        for row in case_rows
                    ]
                )
            )

        for metric_name in metric_names
    }


    return {
        "n_cases":
            len(
                case_rows
            ),

        "aggregation":
            "CASE_LEVEL_THEN_ARITHMETIC_MEAN",

        "case_metrics":
            case_rows,

        "summary":
            summary,
    }


def normal_false_positive_case_metrics(
    segmentation_probability,
    threshold=SEGMENTATION_THRESHOLD,
):
    if float(
        threshold
    ) != SEGMENTATION_THRESHOLD:
        raise ValueError(
            "segmentation threshold is fixed at 0.5."
        )


    probability = np.asarray(
        segmentation_probability,
        dtype=np.float64,
    )


    if probability.size == 0:
        raise ValueError(
            "segmentation probability must be non-empty."
        )


    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "segmentation probabilities must be finite."
        )


    if (
        np.any(
            probability
            <
            0.0
        )
        or
        np.any(
            probability
            >
            1.0
        )
    ):
        raise ValueError(
            "segmentation probabilities must lie in [0, 1]."
        )


    binary = (
        probability
        >=
        float(
            threshold
        )
    )


    positive_pixels = int(
        binary.sum()
    )


    total_pixels = int(
        binary.size
    )


    foreground_fraction = float(
        positive_pixels
        /
        total_pixels
    )


    any_positive = float(
        positive_pixels
        >
        0
    )


    mean_probability = float(
        probability.mean()
    )


    return {
        "predicted_positive_pixels":
            positive_pixels,

        "total_pixels":
            total_pixels,

        "foreground_fraction":
            foreground_fraction,

        "any_positive_case":
            any_positive,

        "mean_segmentation_probability":
            mean_probability,
    }


def evaluate_normal_false_positive(
    segmentation_probabilities,
    threshold=SEGMENTATION_THRESHOLD,
):
    probabilities = np.asarray(
        segmentation_probabilities,
        dtype=np.float64,
    )


    if probabilities.ndim < 2:
        raise ValueError(
            "segmentation probabilities must have shape (N, ...)."
        )


    if probabilities.shape[
        0
    ] == 0:
        raise ValueError(
            "at least one Normal case is required."
        )


    case_rows = [
        normal_false_positive_case_metrics(
            probabilities[
                case_index
            ],
            threshold=threshold,
        )

        for case_index in range(
            probabilities.shape[
                0
            ]
        )
    ]


    summary = {
        "mean_foreground_fraction":
            float(
                np.mean(
                    [
                        row[
                            "foreground_fraction"
                        ]

                        for row in case_rows
                    ]
                )
            ),

        "any_positive_rate":
            float(
                np.mean(
                    [
                        row[
                            "any_positive_case"
                        ]

                        for row in case_rows
                    ]
                )
            ),

        "mean_segmentation_probability":
            float(
                np.mean(
                    [
                        row[
                            "mean_segmentation_probability"
                        ]

                        for row in case_rows
                    ]
                )
            ),
    }


    return {
        "n_cases":
            len(
                case_rows
            ),

        "empty_target_dice_iou":
            "NOT_REPORTED",

        "case_metrics":
            case_rows,

        "summary":
            summary,
    }


def evaluate_external(
    targets,
    diagnostic_probabilities,
    lesion_segmentation_probabilities=None,
    lesion_segmentation_targets=None,
    normal_segmentation_probabilities=None,
    cancer_threshold=CANCER_THRESHOLD,
    segmentation_threshold=SEGMENTATION_THRESHOLD,
):
    diagnostic_metrics, diagnostic_confusion = three_class_metrics(
        targets,
        diagnostic_probabilities,
    )


    cancer_result, cancer_confusion = cancer_metrics(
        targets,
        diagnostic_probabilities,
        threshold=cancer_threshold,
    )


    result = {
        "class_order":
            CLASS_ORDER,

        "thresholds":
            {
                "cancer":
                    float(
                        cancer_threshold
                    ),

                "segmentation":
                    float(
                        segmentation_threshold
                    ),
            },

        "three_class_diagnosis":
            {
                "metrics":
                    diagnostic_metrics,

                "confusion_matrix":
                    diagnostic_confusion.tolist(),
            },

        "malignant_vs_rest_cancer_detection":
            {
                "metrics":
                    cancer_result,

                "confusion_matrix":
                    cancer_confusion,
            },
    }


    if (
        lesion_segmentation_probabilities is None
    ) != (
        lesion_segmentation_targets is None
    ):
        raise ValueError(
            "lesion segmentation probabilities and targets must be supplied together."
        )


    if lesion_segmentation_probabilities is not None:

        result[
            "lesion_segmentation"
        ] = evaluate_lesion_segmentation(
            lesion_segmentation_probabilities,
            lesion_segmentation_targets,
            threshold=segmentation_threshold,
        )


    if normal_segmentation_probabilities is not None:

        result[
            "normal_false_positive"
        ] = evaluate_normal_false_positive(
            normal_segmentation_probabilities,
            threshold=segmentation_threshold,
        )


    return result


__all__ = [
    "CLASS_ORDER",
    "CANCER_THRESHOLD",
    "SEGMENTATION_THRESHOLD",
    "safe_divide",
    "three_class_metrics",
    "cancer_metrics",
    "lesion_segmentation_case_metrics",
    "evaluate_lesion_segmentation",
    "normal_false_positive_case_metrics",
    "evaluate_normal_false_positive",
    "evaluate_external",
]
