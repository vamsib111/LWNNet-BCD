"""
Dependency-aware bootstrap utilities for LWNNet-BCD external evaluation.

Diagnostic bootstrap
--------------------
A single seeded NumPy Generator is created before an ordered sequence of
diagnostic evaluation specifications. The same Generator instance is passed to
successive bootstrap calls, so its state advances across the sequence.

This ordering is scientifically important. Resetting the random seed before
every diagnostic cohort would not reproduce the authenticated evaluation
workflow.

For diagnostic metrics, non-finite bootstrap values are excluded only from the
affected metric distribution. Replicates are not retried. Consequently, valid
bootstrap counts can differ by metric.

Case-metric bootstrap
---------------------
Finite case-level metric vectors are resampled by dependency unit using the
authenticated fixed study settings. Each call initializes the frozen study
seed, generates exactly 5000 bootstrap means, and requires every value to be
finite.

Study settings
--------------
Bootstrap replicates: 5000
Seed: 20260830
Percentile CI: 2.5 to 97.5

Historical external diagnostic authority SHA256:
    0b9af4680512306b7da919876b2293d8d8e436632804876c9f6b401019bcc531

Historical external segmentation authority SHA256:
    fd5dc664c0bb6cd237c458dcafef9a10e62ca5bf8e1ce13762a8676a24ce4d12

Authenticated helper SHA256 values:
    diagnostic bootstrap_endpoint:
        29649849e78c3950e3c9384dc403d6c4aa448e5c194ed816d7e24d3413d9ca83
    diagnostic percentile_ci:
        a49a820d65250dddba978a643f1b48bb00ad7fed0c1d962b29cba5724a154686
    case bootstrap_case_means:
        99b6eef7ca78806e11bddcad11b5d41a6a8f913918418a57945b9266f3fa34f3
    case percentile_ci:
        d8cef657726a58cdce91841065029a1a85ac8aff22de25987f034b081fa68dc5
"""

import math

import numpy as np

from .evaluate_external import (
    CANCER_THRESHOLD,
    cancer_metrics,
    three_class_metrics,
)


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260830

LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5


def _validate_groups(
    groups,
    expected_length,
):
    groups = np.asarray(
        groups,
        dtype=object,
    ).reshape(
        -1
    )


    if groups.size != expected_length:
        raise ValueError(
            "bootstrap group count must match the number of cases."
        )


    group_strings = np.asarray(
        [
            str(
                group
            )

            for group in groups
        ],
        dtype=object,
    )


    unique_groups = sorted(
        set(
            group_strings.tolist()
        )
    )


    if not unique_groups:
        raise ValueError(
            "bootstrap cohort contains no dependency units."
        )


    group_to_indices = {
        group:
            np.where(
                group_strings
                ==
                group
            )[
                0
            ]

        for group in unique_groups
    }


    return (
        unique_groups,
        group_to_indices,
    )


def bootstrap_endpoint(
    y_true,
    probabilities,
    groups,
    endpoint,
    threshold,
    replicates,
    rng,
):
    """
    Bootstrap one diagnostic endpoint using a caller-owned RNG stream.

    The caller must preserve RNG state across ordered specifications when exact
    study-workflow reproduction is required.
    """

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


    if probabilities.shape != (
        y_true.size,
        3,
    ):
        raise ValueError(
            "probabilities must have shape (N, 3)."
        )


    if int(
        replicates
    ) <= 0:
        raise ValueError(
            "replicates must be positive."
        )


    (
        unique_groups,
        group_to_indices,
    ) = _validate_groups(
        groups,
        y_true.size,
    )


    if endpoint == "THREE_CLASS":

        point_metrics, _ = three_class_metrics(
            y_true,
            probabilities,
        )


    elif endpoint == "MALIGNANT_VS_REST":

        point_metrics, _ = cancer_metrics(
            y_true,
            probabilities,
            threshold,
        )


    else:

        raise ValueError(
            "endpoint must be THREE_CLASS or MALIGNANT_VS_REST."
        )


    bootstrap_values = {
        metric_name:
            []

        for metric_name in point_metrics
    }


    for _ in range(
        int(
            replicates
        )
    ):

        sampled_group_positions = rng.integers(
            low=0,
            high=len(
                unique_groups
            ),
            size=len(
                unique_groups
            ),
        )


        sampled_indices = np.concatenate(
            [
                group_to_indices[
                    unique_groups[
                        int(
                            position
                        )
                    ]
                ]

                for position in sampled_group_positions
            ]
        )


        sampled_y = y_true[
            sampled_indices
        ]


        sampled_probabilities = probabilities[
            sampled_indices
        ]


        if endpoint == "THREE_CLASS":

            sampled_metrics, _ = three_class_metrics(
                sampled_y,
                sampled_probabilities,
            )


        else:

            sampled_metrics, _ = cancer_metrics(
                sampled_y,
                sampled_probabilities,
                threshold,
            )


        for metric_name, value in sampled_metrics.items():

            if math.isfinite(
                float(
                    value
                )
            ):

                bootstrap_values[
                    metric_name
                ].append(
                    float(
                        value
                    )
                )


    return (
        point_metrics,
        bootstrap_values,
        len(
            unique_groups
        ),
    )


def diagnostic_percentile_ci(
    values,
):
    """
    Percentile CI for one finite-only diagnostic bootstrap distribution.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    if values.size == 0:

        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )


    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "diagnostic percentile input must contain only finite values."
        )


    return (
        float(
            np.percentile(
                values,
                LOWER_PERCENTILE,
            )
        ),
        float(
            np.percentile(
                values,
                UPPER_PERCENTILE,
            )
        ),
    )


def _summarize_diagnostic_result(
    point_metrics,
    bootstrap_values,
    bootstrap_group_count,
    replicates,
):
    metrics = {}


    for metric_name, values in bootstrap_values.items():

        lower, upper = diagnostic_percentile_ci(
            values
        )


        metrics[
            metric_name
        ] = {
            "estimate":
                float(
                    point_metrics[
                        metric_name
                    ]
                ),

            "CI_lower_95":
                lower,

            "CI_upper_95":
                upper,

            "valid_bootstrap_replicates":
                int(
                    len(
                        values
                    )
                ),

            "total_bootstrap_replicates":
                int(
                    replicates
                ),
        }


    return {
        "bootstrap_unit_count":
            int(
                bootstrap_group_count
            ),

        "metrics":
            metrics,

        "bootstrap_values":
            bootstrap_values,
    }


def run_diagnostic_bootstrap_sequence(
    specifications,
    seed=BOOTSTRAP_SEED,
    replicates=BOOTSTRAP_REPLICATES,
):
    """
    Run an ORDERED diagnostic-bootstrap sequence with one shared RNG stream.

    Each specification must contain:
        name
        y_true
        probabilities
        groups
        endpoint

    An optional threshold may be supplied. If omitted, 0.5 is used.

    Specification order changes later bootstrap draws because the shared RNG
    state advances after every preceding specification.
    """

    specifications = list(
        specifications
    )


    if not specifications:
        raise ValueError(
            "at least one diagnostic bootstrap specification is required."
        )


    if int(
        replicates
    ) <= 0:
        raise ValueError(
            "replicates must be positive."
        )


    rng = np.random.default_rng(
        int(
            seed
        )
    )


    results = []


    for sequence_index, specification in enumerate(
        specifications
    ):

        required = {
            "name",
            "y_true",
            "probabilities",
            "groups",
            "endpoint",
        }


        missing = (
            required
            -
            set(
                specification
            )
        )


        if missing:

            raise ValueError(
                "diagnostic bootstrap specification is missing: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )


        threshold = float(
            specification.get(
                "threshold",
                CANCER_THRESHOLD,
            )
        )


        (
            point_metrics,
            bootstrap_values,
            group_count,
        ) = bootstrap_endpoint(
            y_true=specification[
                "y_true"
            ],
            probabilities=specification[
                "probabilities"
            ],
            groups=specification[
                "groups"
            ],
            endpoint=specification[
                "endpoint"
            ],
            threshold=threshold,
            replicates=replicates,
            rng=rng,
        )


        summary = _summarize_diagnostic_result(
            point_metrics,
            bootstrap_values,
            group_count,
            replicates,
        )


        summary[
            "sequence_index"
        ] = int(
            sequence_index
        )


        summary[
            "name"
        ] = str(
            specification[
                "name"
            ]
        )


        summary[
            "endpoint"
        ] = str(
            specification[
                "endpoint"
            ]
        )


        summary[
            "threshold"
        ] = threshold


        results.append(
            summary
        )


    return {
        "bootstrap_seed":
            int(
                seed
            ),

        "bootstrap_replicates":
            int(
                replicates
            ),

        "bootstrap_method":
            "PERCENTILE",

        "percentile_bounds":
            (
                LOWER_PERCENTILE,
                UPPER_PERCENTILE,
            ),

        "rng_policy":
            "ONE_SHARED_STREAM_ACROSS_ORDERED_SPECIFICATIONS",

        "sequence_order_is_significant":
            True,

        "results":
            results,
    }


def bootstrap_case_means(
    metric_matrix,
    group_ids,
):
    """
    Bootstrap finite case-level metrics using the authenticated study settings.

    Unlike the diagnostic ordered-sequence workflow, this historical helper
    initializes the frozen seed inside each call.
    """

    metric_matrix = np.asarray(
        metric_matrix,
        dtype=np.float64,
    )


    group_ids = np.asarray(
        group_ids,
        dtype=object,
    ).reshape(
        -1
    )


    if metric_matrix.ndim != 2:
        raise ValueError(
            "metric_matrix must be two-dimensional."
        )


    if metric_matrix.shape[
        0
    ] != group_ids.size:
        raise ValueError(
            "metric/group length mismatch."
        )


    if not np.isfinite(
        metric_matrix
    ).all():
        raise ValueError(
            "case metrics must be finite before bootstrap."
        )


    (
        unique_groups,
        group_to_indices,
    ) = _validate_groups(
        group_ids,
        metric_matrix.shape[
            0
        ],
    )


    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )


    bootstrap_values = np.empty(
        (
            BOOTSTRAP_REPLICATES,
            metric_matrix.shape[
                1
            ],
        ),
        dtype=np.float64,
    )


    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):

        sampled_group_positions = rng.integers(
            low=0,
            high=len(
                unique_groups
            ),
            size=len(
                unique_groups
            ),
        )


        sampled_indices = np.concatenate(
            [
                group_to_indices[
                    unique_groups[
                        int(
                            position
                        )
                    ]
                ]

                for position in sampled_group_positions
            ]
        )


        bootstrap_values[
            replicate
        ] = metric_matrix[
            sampled_indices
        ].mean(
            axis=0
        )


    if not np.isfinite(
        bootstrap_values
    ).all():
        raise ValueError(
            "case-metric bootstrap produced a non-finite value."
        )


    return (
        bootstrap_values,
        len(
            unique_groups
        ),
    )


def case_mean_percentile_ci(
    values,
):
    """
    Percentile CI requiring exactly 5000 finite case-metric replicates.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    if (
        values.ndim != 1
        or
        values.size != BOOTSTRAP_REPLICATES
        or
        not np.isfinite(
            values
        ).all()
    ):

        raise ValueError(
            "case-metric bootstrap CI requires exactly 5000 finite values."
        )


    return (
        float(
            np.percentile(
                values,
                LOWER_PERCENTILE,
            )
        ),
        float(
            np.percentile(
                values,
                UPPER_PERCENTILE,
            )
        ),
    )


def summarize_case_metric_bootstrap(
    metric_matrix,
    group_ids,
    metric_names=None,
):
    """
    Summarize finite case-level metric vectors using the frozen 5000-replicate
    dependency-unit bootstrap.
    """

    metric_matrix = np.asarray(
        metric_matrix,
        dtype=np.float64,
    )


    if metric_matrix.ndim != 2:
        raise ValueError(
            "metric_matrix must be two-dimensional."
        )


    if metric_names is None:

        metric_names = [
            f"metric_{index}"

            for index in range(
                metric_matrix.shape[
                    1
                ]
            )
        ]


    else:

        metric_names = [
            str(
                name
            )

            for name in metric_names
        ]


    if len(
        metric_names
    ) != metric_matrix.shape[
        1
    ]:

        raise ValueError(
            "metric_names length must match metric_matrix columns."
        )


    (
        bootstrap_values,
        group_count,
    ) = bootstrap_case_means(
        metric_matrix,
        group_ids,
    )


    point_estimates = metric_matrix.mean(
        axis=0
    )


    summary = {}


    for metric_index, metric_name in enumerate(
        metric_names
    ):

        lower, upper = case_mean_percentile_ci(
            bootstrap_values[
                :,
                metric_index
            ]
        )


        summary[
            metric_name
        ] = {
            "estimate":
                float(
                    point_estimates[
                        metric_index
                    ]
                ),

            "CI_lower_95":
                lower,

            "CI_upper_95":
                upper,

            "valid_bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "total_bootstrap_replicates":
                BOOTSTRAP_REPLICATES,
        }


    return {
        "bootstrap_unit_count":
            int(
                group_count
            ),

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "bootstrap_method":
            "PERCENTILE",

        "percentile_bounds":
            (
                LOWER_PERCENTILE,
                UPPER_PERCENTILE,
            ),

        "metrics":
            summary,

        "bootstrap_values":
            bootstrap_values,
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "LOWER_PERCENTILE",
    "UPPER_PERCENTILE",
    "bootstrap_endpoint",
    "diagnostic_percentile_ci",
    "run_diagnostic_bootstrap_sequence",
    "bootstrap_case_means",
    "case_mean_percentile_ci",
    "summarize_case_metric_bootstrap",
]
