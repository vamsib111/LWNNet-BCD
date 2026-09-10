# ==================================================================================================
# LWNNet-BCD
# FROZEN R4 HIERARCHICAL TRAINING OBJECTIVE
#
# Frozen by R4-01R1.
#
# Classes:
#     0 = Normal
#     1 = Benign
#     2 = Malignant
#
# Diagnostic hierarchy:
#
#     q_L = P(lesion present | image)
#     q_C = P(malignant | lesion, image)
#
#     L_diag =
#         0.5 * L_presence
#       + 0.5 * L_malignancy_given_lesion
#
# Joint objective:
#
#     L_total =
#         1.00 * L_segmentation
#       + 0.25 * L_boundary
#       + 0.50 * L_diagnosis
#
# Numeric pixel-positive weights must be calculated separately for EACH fold
# from that fold's TRAINING masks only.
# ==================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def moderated_positive_pixel_weight(
    negative_pixels,
    positive_pixels,
):
    negative_pixels = float(
        negative_pixels
    )

    positive_pixels = float(
        positive_pixels
    )

    if negative_pixels <= 0.0:
        raise ValueError(
            "negative_pixels must be > 0"
        )

    if positive_pixels <= 0.0:
        raise ValueError(
            "positive_pixels must be > 0"
        )

    return float(
        (
            negative_pixels
            /
            positive_pixels
        )
        ** 0.5
    )


def positive_image_soft_dice_loss(
    logits,
    targets,
    epsilon=1e-6,
):
    targets = targets.float()

    positive_image_mask = (
        targets.flatten(
            1
        ).sum(
            dim=1
        )
        >
        0
    )

    if not torch.any(
        positive_image_mask
    ):
        return logits.sum() * 0.0

    selected_logits = logits[
        positive_image_mask
    ]

    selected_targets = targets[
        positive_image_mask
    ].float()

    probabilities = torch.sigmoid(
        selected_logits.float()
    )

    intersection = (
        probabilities
        *
        selected_targets
    ).flatten(
        1
    ).sum(
        dim=1
    )

    probability_mass = probabilities.flatten(
        1
    ).sum(
        dim=1
    )

    target_mass = selected_targets.flatten(
        1
    ).sum(
        dim=1
    )

    dice = (
        2.0
        *
        intersection
        +
        float(
            epsilon
        )
    ) / (
        probability_mass
        +
        target_mass
        +
        float(
            epsilon
        )
    )

    return (
        1.0
        -
        dice.mean()
    )


def balanced_binary_bce_with_logits(
    logits,
    targets,
    negative_weight,
    positive_weight,
):
    logits = logits.float()

    targets = targets.float()

    raw_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    negative_weight_tensor = torch.as_tensor(
        float(
            negative_weight
        ),
        dtype=raw_loss.dtype,
        device=raw_loss.device,
    )

    positive_weight_tensor = torch.as_tensor(
        float(
            positive_weight
        ),
        dtype=raw_loss.dtype,
        device=raw_loss.device,
    )

    sample_weights = torch.where(
        targets > 0.5,
        positive_weight_tensor,
        negative_weight_tensor,
    )

    return (
        raw_loss
        *
        sample_weights
    ).mean()


class R4LocalizationLoss(
    nn.Module
):

    def __init__(
        self,
        segmentation_positive_weight,
        boundary_positive_weight,
        epsilon=1e-6,
    ):

        super().__init__()

        self.register_buffer(
            "segmentation_positive_weight",
            torch.tensor(
                float(
                    segmentation_positive_weight
                ),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "boundary_positive_weight",
            torch.tensor(
                float(
                    boundary_positive_weight
                ),
                dtype=torch.float32,
            ),
        )

        self.epsilon = float(
            epsilon
        )


    def forward(
        self,
        segmentation_logits,
        boundary_logits,
        segmentation_targets,
        boundary_targets,
    ):

        segmentation_targets = segmentation_targets.float()

        boundary_targets = boundary_targets.float()


        segmentation_bce = F.binary_cross_entropy_with_logits(
            segmentation_logits.float(),
            segmentation_targets,
            pos_weight=self.segmentation_positive_weight,
        )


        segmentation_dice = positive_image_soft_dice_loss(
            logits=segmentation_logits,
            targets=segmentation_targets,
            epsilon=self.epsilon,
        )


        segmentation_loss = (
            0.50
            *
            segmentation_bce
            +
            0.50
            *
            segmentation_dice
        )


        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits.float(),
            boundary_targets,
            pos_weight=self.boundary_positive_weight,
        )


        boundary_dice = positive_image_soft_dice_loss(
            logits=boundary_logits,
            targets=boundary_targets,
            epsilon=self.epsilon,
        )


        boundary_loss = (
            0.50
            *
            boundary_bce
            +
            0.50
            *
            boundary_dice
        )


        return {
            "segmentation_loss":
                segmentation_loss,

            "segmentation_bce":
                segmentation_bce,

            "segmentation_positive_image_dice_loss":
                segmentation_dice,

            "boundary_loss":
                boundary_loss,

            "boundary_bce":
                boundary_bce,

            "boundary_positive_image_dice_loss":
                boundary_dice,
        }


class R4HierarchicalDiagnosticLoss(
    nn.Module
):

    def __init__(
        self,
        presence_negative_weight,
        presence_positive_weight,
        malignancy_negative_weight,
        malignancy_positive_weight,
    ):

        super().__init__()

        self.presence_negative_weight = float(
            presence_negative_weight
        )

        self.presence_positive_weight = float(
            presence_positive_weight
        )

        self.malignancy_negative_weight = float(
            malignancy_negative_weight
        )

        self.malignancy_positive_weight = float(
            malignancy_positive_weight
        )


    def forward(
        self,
        outputs,
        class_targets,
    ):

        class_targets = class_targets.long()


        if class_targets.ndim != 1:

            raise ValueError(
                "class_targets must have shape [B]"
            )


        if not torch.all(
            (
                class_targets >= 0
            )
            &
            (
                class_targets <= 2
            )
        ):

            raise ValueError(
                "class_targets must use 0=Normal, 1=Benign, 2=Malignant"
            )


        presence_targets = (
            class_targets
            !=
            0
        ).float().unsqueeze(
            1
        )


        presence_loss = balanced_binary_bce_with_logits(
            logits=outputs[
                "lesion_presence_logit"
            ],
            targets=presence_targets,
            negative_weight=self.presence_negative_weight,
            positive_weight=self.presence_positive_weight,
        )


        lesion_mask = (
            class_targets
            !=
            0
        )


        if not torch.any(
            lesion_mask
        ):

            malignancy_loss = (
                outputs[
                    "malignancy_given_lesion_logit"
                ].sum()
                *
                0.0
            )


        else:

            malignancy_targets = (
                class_targets[
                    lesion_mask
                ]
                ==
                2
            ).float().unsqueeze(
                1
            )


            malignancy_loss = balanced_binary_bce_with_logits(
                logits=outputs[
                    "malignancy_given_lesion_logit"
                ][
                    lesion_mask
                ],
                targets=malignancy_targets,
                negative_weight=self.malignancy_negative_weight,
                positive_weight=self.malignancy_positive_weight,
            )


        diagnosis_loss = (
            0.50
            *
            presence_loss
            +
            0.50
            *
            malignancy_loss
        )


        class_probs = outputs[
            "diagnostic_class_probs"
        ].float()


        true_class_probability = class_probs[
            torch.arange(
                class_targets.shape[
                    0
                ],
                device=class_targets.device,
            ),
            class_targets,
        ]


        # Monitoring only.
        # NOT added to the optimization objective.
        monitoring_hierarchical_nll = (
            -
            torch.log(
                true_class_probability.clamp_min(
                    1e-7
                )
            )
        ).mean()


        return {
            "diagnosis_loss":
                diagnosis_loss,

            "presence_loss":
                presence_loss,

            "malignancy_given_lesion_loss":
                malignancy_loss,

            "monitoring_hierarchical_nll":
                monitoring_hierarchical_nll,
        }


class R4JointObjective(
    nn.Module
):

    def __init__(
        self,
        segmentation_positive_weight,
        boundary_positive_weight,
        presence_negative_weight,
        presence_positive_weight,
        malignancy_negative_weight,
        malignancy_positive_weight,
        lambda_segmentation=1.00,
        lambda_boundary=0.25,
        lambda_diagnosis=0.50,
    ):

        super().__init__()


        self.localization_loss = R4LocalizationLoss(
            segmentation_positive_weight=segmentation_positive_weight,
            boundary_positive_weight=boundary_positive_weight,
        )


        self.diagnostic_loss = R4HierarchicalDiagnosticLoss(
            presence_negative_weight=presence_negative_weight,
            presence_positive_weight=presence_positive_weight,
            malignancy_negative_weight=malignancy_negative_weight,
            malignancy_positive_weight=malignancy_positive_weight,
        )


        self.lambda_segmentation = float(
            lambda_segmentation
        )


        self.lambda_boundary = float(
            lambda_boundary
        )


        self.lambda_diagnosis = float(
            lambda_diagnosis
        )


    def forward(
        self,
        outputs,
        segmentation_targets,
        boundary_targets,
        class_targets,
    ):

        localization = self.localization_loss(
            segmentation_logits=outputs[
                "segmentation_logits"
            ],
            boundary_logits=outputs[
                "boundary_logits"
            ],
            segmentation_targets=segmentation_targets,
            boundary_targets=boundary_targets,
        )


        diagnosis = self.diagnostic_loss(
            outputs=outputs,
            class_targets=class_targets,
        )


        total_loss = (
            self.lambda_segmentation
            *
            localization[
                "segmentation_loss"
            ]
            +
            self.lambda_boundary
            *
            localization[
                "boundary_loss"
            ]
            +
            self.lambda_diagnosis
            *
            diagnosis[
                "diagnosis_loss"
            ]
        )


        return {
            "total_loss":
                total_loss,

            **localization,

            **diagnosis,
        }
