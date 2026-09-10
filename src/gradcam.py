"""
Native Grad-CAM for LWNNet-BCD.

Frozen study target layer:
    diagnostic_feature_adapter.projection

Track A:
    native probability of the model's original predicted class.

Track B:
    native P(Malignant), exposed through cancer_probability and equivalent to
    diagnostic_class_probs[:, 2].

The model outputs native normalized diagnostic probabilities. No additional
softmax or sigmoid is applied.

Grad-CAM:
    spatial mean of target-score gradients
    -> channel-weighted activation sum
    -> ReLU
    -> bilinear resize to 128 x 128 with align_corners=False
    -> min-max normalization

Corrected historical implementation authorities:
    d36f6471a591133944d5dd38c22d720aa054dba1ccb2e9cb6fb01181c26b7346
    8bc0915d5b24d34ef4e08d74ce702cea9eced011d3f5a9491f7fac3689cbd40c
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_ORDER = (
    "Normal",
    "Benign",
    "Malignant",
)

MALIGNANT_INDEX = 2

TARGET_MODULE_PATH = (
    "diagnostic_feature_adapter.projection"
)

EXPECTED_ACTIVATION_SHAPE = (
    1,
    96,
    16,
    16,
)

CAM_OUTPUT_SIZE = (
    128,
    128,
)

CAM_ZERO_EPSILON = 1e-12


def _resolve_target_module(
    model,
):
    named_modules = dict(
        model.named_modules()
    )

    if TARGET_MODULE_PATH not in named_modules:

        raise ValueError(
            "Frozen Grad-CAM target module is absent from the model."
        )

    target_module = named_modules[
        TARGET_MODULE_PATH
    ]

    if not isinstance(
        target_module,
        nn.Conv2d,
    ):

        raise TypeError(
            "Frozen Grad-CAM target must be a Conv2d module."
        )

    return target_module


def normalized_gradcam(
    activation,
    gradient,
):
    if (
        not torch.is_tensor(
            activation
        )
        or
        not torch.is_tensor(
            gradient
        )
    ):

        raise TypeError(
            "activation and gradient must be tensors."
        )

    if (
        activation.ndim != 4
        or
        gradient.ndim != 4
    ):

        raise ValueError(
            "activation and gradient must use NCHW layout."
        )

    if tuple(
        activation.shape
    ) != tuple(
        gradient.shape
    ):

        raise ValueError(
            "activation and gradient shapes must match."
        )

    if activation.shape[0] != 1:

        raise ValueError(
            "Grad-CAM expects a single-image batch."
        )

    weights = gradient.mean(
        dim=(
            2,
            3,
        ),
        keepdim=True,
    )

    raw_cam = (
        weights
        *
        activation
    ).sum(
        dim=1,
        keepdim=True,
    )

    raw_cam = F.relu(
        raw_cam
    )

    resized_cam = F.interpolate(
        raw_cam,
        size=CAM_OUTPUT_SIZE,
        mode="bilinear",
        align_corners=False,
    )

    cam = resized_cam[
        0,
        0,
    ]

    cam_min = cam.min()

    cam_max = cam.max()

    cam_range = (
        cam_max
        -
        cam_min
    )

    zero_cam = (
        float(
            cam_range
            .detach()
            .cpu()
            .item()
        )
        <=
        CAM_ZERO_EPSILON
    )

    if zero_cam:

        normalized = torch.zeros_like(
            cam
        )

    else:

        normalized = (
            cam
            -
            cam_min
        ) / cam_range

    normalized_np = (
        normalized
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    if not np.isfinite(
        normalized_np
    ).all():

        raise RuntimeError(
            "normalized Grad-CAM contains non-finite values."
        )

    return (
        normalized_np,
        bool(
            zero_cam
        ),
    )


def generate_native_gradcams(
    model,
    input_tensor,
    include_malignant_track=False,
):
    if model.training:

        raise ValueError(
            "Grad-CAM requires model.eval() mode."
        )

    if not torch.is_tensor(
        input_tensor
    ):

        raise TypeError(
            "input_tensor must be a tensor."
        )

    if tuple(
        input_tensor.shape
    ) != (
        1,
        3,
        128,
        128,
    ):

        raise ValueError(
            "input_tensor must have shape (1, 3, 128, 128)."
        )

    if not torch.is_floating_point(
        input_tensor
    ):

        raise TypeError(
            "input_tensor must be floating point."
        )

    target_module = _resolve_target_module(
        model
    )

    activation_holder = {
        "tensor":
            None,
    }

    def activation_hook(
        module,
        inputs,
        output,
    ):
        del module
        del inputs

        if not torch.is_tensor(
            output
        ):

            raise TypeError(
                "Grad-CAM activation must be a tensor."
            )

        activation_holder[
            "tensor"
        ] = output

    hook_handle = target_module.register_forward_hook(
        activation_hook
    )

    try:

        with torch.enable_grad():

            output = model(
                input_tensor
            )

            if not isinstance(
                output,
                dict,
            ):

                raise TypeError(
                    "LWNNet-BCD model output must be a dictionary."
                )

            required = {
                "diagnostic_class_probs",
                "cancer_probability",
            }

            missing = (
                required
                -
                set(
                    output
                )
            )

            if missing:

                raise KeyError(
                    "required diagnostic output missing: "
                    +
                    ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

            probabilities = output[
                "diagnostic_class_probs"
            ]

            cancer_probability = output[
                "cancer_probability"
            ].reshape(
                -1
            )

            if tuple(
                probabilities.shape
            ) != (
                1,
                3,
            ):

                raise ValueError(
                    "diagnostic_class_probs must have shape (1, 3)."
                )

            if cancer_probability.numel() != 1:

                raise ValueError(
                    "cancer_probability must contain one value."
                )

            probability_sum_error = abs(
                float(
                    probabilities
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                -
                1.0
            )

            if probability_sum_error > 1e-5:

                raise ValueError(
                    "diagnostic_class_probs are not normalized."
                )

            activation = activation_holder[
                "tensor"
            ]

            if activation is None:

                raise RuntimeError(
                    "Grad-CAM activation hook did not fire."
                )

            if tuple(
                activation.shape
            ) != EXPECTED_ACTIVATION_SHAPE:

                raise ValueError(
                    "Grad-CAM activation shape changed."
                )

            predicted_index = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                )[0].item()
            )

            track_a_score = probabilities[
                0,
                predicted_index
            ]

            track_a_gradient = torch.autograd.grad(
                outputs=track_a_score,
                inputs=activation,
                retain_graph=bool(
                    include_malignant_track
                ),
                create_graph=False,
                allow_unused=False,
            )[0]

            track_a_cam, track_a_zero = normalized_gradcam(
                activation,
                track_a_gradient,
            )

            track_b = None

            if include_malignant_track:

                native_cancer_probability = cancer_probability[
                    0
                ]

                malignant_probability = probabilities[
                    0,
                    MALIGNANT_INDEX
                ]

                if not torch.allclose(
                    native_cancer_probability.detach(),
                    malignant_probability.detach(),
                    atol=1e-7,
                    rtol=1e-6,
                ):

                    raise RuntimeError(
                        "cancer_probability differs from native P(Malignant)."
                    )

                track_b_gradient = torch.autograd.grad(
                    outputs=native_cancer_probability,
                    inputs=activation,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                track_b_cam, track_b_zero = normalized_gradcam(
                    activation,
                    track_b_gradient,
                )

                track_b = {
                    "target":
                        "native_malignant_probability",

                    "target_class_index":
                        MALIGNANT_INDEX,

                    "target_class":
                        CLASS_ORDER[
                            MALIGNANT_INDEX
                        ],

                    "target_probability":
                        float(
                            native_cancer_probability
                            .detach()
                            .cpu()
                            .item()
                        ),

                    "cam":
                        track_b_cam,

                    "zero_cam":
                        bool(
                            track_b_zero
                        ),
                }

            probabilities_list = [
                float(
                    probabilities[
                        0,
                        class_index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

                for class_index in range(
                    3
                )
            ]

            return {
                "class_order":
                    CLASS_ORDER,

                "probabilities":
                    probabilities_list,

                "predicted_class_index":
                    predicted_index,

                "predicted_class":
                    CLASS_ORDER[
                        predicted_index
                    ],

                "target_module":
                    TARGET_MODULE_PATH,

                "track_a":
                    {
                        "target":
                            "native_original_predicted_class_probability",

                        "target_class_index":
                            predicted_index,

                        "target_class":
                            CLASS_ORDER[
                                predicted_index
                            ],

                        "target_probability":
                            float(
                                track_a_score
                                .detach()
                                .cpu()
                                .item()
                            ),

                        "cam":
                            track_a_cam,

                        "zero_cam":
                            bool(
                                track_a_zero
                            ),
                    },

                "track_b":
                    track_b,
            }

    finally:

        hook_handle.remove()


def generate_predicted_class_gradcam(
    model,
    input_tensor,
):
    return generate_native_gradcams(
        model=model,
        input_tensor=input_tensor,
        include_malignant_track=False,
    )[
        "track_a"
    ]


def generate_predicted_and_malignant_gradcams(
    model,
    input_tensor,
):
    return generate_native_gradcams(
        model=model,
        input_tensor=input_tensor,
        include_malignant_track=True,
    )


__all__ = [
    "CLASS_ORDER",
    "MALIGNANT_INDEX",
    "TARGET_MODULE_PATH",
    "EXPECTED_ACTIVATION_SHAPE",
    "CAM_OUTPUT_SIZE",
    "CAM_ZERO_EPSILON",
    "normalized_gradcam",
    "generate_native_gradcams",
    "generate_predicted_class_gradcam",
    "generate_predicted_and_malignant_gradcams",
]
