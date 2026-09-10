# ==================================================================================================
# LWNNet-BCD
# FROZEN R4 BOUNDARY-TARGET RUNTIME
#
# Exact historical R3 create_boundary_target recovered by R4-02R2R1.
#
# Method:
# binary horizontal/vertical transitions followed by 3x3 max-pool thickening.
# ==================================================================================================

import torch
import torch.nn.functional as F

BOUNDARY_THICKENING_KERNEL = 3


def create_boundary_target(
    masks,
    thickening_kernel=3,
):

    if masks.ndim != 4:

        raise ValueError(
            "Boundary target expects masks with shape [B,1,H,W]."
        )


    if masks.shape[
        1
    ] != 1:

        raise ValueError(
            "Boundary target expects exactly one mask channel."
        )


    if thickening_kernel < 1:

        raise ValueError(
            "Boundary thickening kernel must be >= 1."
        )


    if thickening_kernel % 2 == 0:

        raise ValueError(
            "Boundary thickening kernel must be odd."
        )


    binary = (
        masks
        >=
        0.5
    ).float()


    boundary = torch.zeros_like(
        binary
    )


    # Horizontal transitions.
    horizontal_transition = (
        binary[
            :,
            :,
            1:,
            :
        ]
        !=
        binary[
            :,
            :,
            :-1,
            :
        ]
    )


    boundary[
        :,
        :,
        1:,
        :
    ] = torch.maximum(
        boundary[
            :,
            :,
            1:,
            :
        ],
        horizontal_transition.float(),
    )


    boundary[
        :,
        :,
        :-1,
        :
    ] = torch.maximum(
        boundary[
            :,
            :,
            :-1,
            :
        ],
        horizontal_transition.float(),
    )


    # Vertical transitions.
    vertical_transition = (
        binary[
            :,
            :,
            :,
            1:
        ]
        !=
        binary[
            :,
            :,
            :,
            :-1
        ]
    )


    boundary[
        :,
        :,
        :,
        1:
    ] = torch.maximum(
        boundary[
            :,
            :,
            :,
            1:
        ],
        vertical_transition.float(),
    )


    boundary[
        :,
        :,
        :,
        :-1
    ] = torch.maximum(
        boundary[
            :,
            :,
            :,
            :-1
        ],
        vertical_transition.float(),
    )


    if thickening_kernel > 1:

        padding = (
            thickening_kernel
            //
            2
        )


        boundary = F.max_pool2d(
            boundary,
            kernel_size=thickening_kernel,
            stride=1,
            padding=padding,
        )


    return (
        boundary
        >
        0
    ).float()
