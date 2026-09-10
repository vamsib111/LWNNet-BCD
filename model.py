# ==================================================================================================
# LWNNet-BCD
# CANONICAL AUTHENTICATED R4 RUNTIME
#
# Recovered from exact IPython execution history by R4-P01R2.
#
# Frozen authority:
#     R3 parameters       : 209018
#     R4 parameters       : 210684
#     R4 signature        : a738a385080ce19aef6856c4d917bf75cca95183d38f95d09a17d64766b65917
#
# Architecture/runtime source only.
# No training, data loading, evaluation or holdout access code.
# ==================================================================================================

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASTER_SEED = 20260830



class R3DepthwiseSeparableBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        groups=8,
    ):

        super().__init__()


        if in_channels % groups != 0:

            raise ValueError(
                f"in_channels={in_channels} must be divisible by groups={groups}."
            )


        if out_channels % groups != 0:

            raise ValueError(
                f"out_channels={out_channels} must be divisible by groups={groups}."
            )


        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )


        self.depthwise_norm = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
        )


        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )


        self.pointwise_norm = nn.GroupNorm(
            num_groups=groups,
            num_channels=out_channels,
        )


        self.activation = nn.SiLU(
            inplace=True
        )


    def forward(
        self,
        x
    ):

        x = self.depthwise(
            x
        )

        x = self.depthwise_norm(
            x
        )

        x = self.activation(
            x
        )


        x = self.pointwise(
            x
        )

        x = self.pointwise_norm(
            x
        )

        x = self.activation(
            x
        )


        return x


class R3EfficientChannelAttention(
    nn.Module
):

    def __init__(
        self,
        kernel_size=3
    ):

        super().__init__()


        if kernel_size % 2 == 0:

            raise ValueError(
                "ECA kernel size must be odd."
            )


        self.global_pool = nn.AdaptiveAvgPool2d(
            1
        )


        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(
                kernel_size
                -
                1
            )
            //
            2,
            bias=False,
        )


        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        x
    ):

        attention = self.global_pool(
            x
        )


        attention = (
            attention
            .squeeze(
                -1
            )
            .transpose(
                -1,
                -2
            )
        )


        attention = self.channel_conv(
            attention
        )


        attention = (
            attention
            .transpose(
                -1,
                -2
            )
            .unsqueeze(
                -1
            )
        )


        attention = self.sigmoid(
            attention
        )


        return (
            x
            *
            attention
        )


class R3LightweightMultiScaleContext(
    nn.Module
):

    def __init__(
        self,
        in_channels=160,
        branch_channels=64,
        out_channels=192,
    ):

        super().__init__()


        self.branch_d1 = R3DepthwiseSeparableBlock(
            in_channels=in_channels,
            out_channels=branch_channels,
            dilation=1,
        )


        self.branch_d2 = R3DepthwiseSeparableBlock(
            in_channels=in_channels,
            out_channels=branch_channels,
            dilation=2,
        )


        self.branch_d4 = R3DepthwiseSeparableBlock(
            in_channels=in_channels,
            out_channels=branch_channels,
            dilation=4,
        )


        fused_channels = (
            3
            *
            branch_channels
        )


        self.fusion = nn.Sequential(

            nn.Conv2d(
                in_channels=fused_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.SiLU(
                inplace=True
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


    def forward(
        self,
        x
    ):

        branch_1 = self.branch_d1(
            x
        )

        branch_2 = self.branch_d2(
            x
        )

        branch_4 = self.branch_d4(
            x
        )


        fused = torch.cat(
            [
                branch_1,
                branch_2,
                branch_4,
            ],
            dim=1,
        )


        return self.fusion(
            fused
        )


class LWNNetBCDReviewerUpgradeCandidate(
    nn.Module
):

    def __init__(
        self,
        num_diagnostic_classes=3
    ):

        super().__init__()


        # ------------------------------------------------------------------------------------------
        # Stem
        # ------------------------------------------------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=32,
            ),

            nn.SiLU(
                inplace=True
            ),
        )


        # ------------------------------------------------------------------------------------------
        # Shared hierarchical encoder
        # 128 -> 64 -> 32 -> 16
        # ------------------------------------------------------------------------------------------

        self.encoder_stage1 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        self.encoder_stage2 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=64,
                out_channels=96,
                stride=2,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        self.encoder_stage3 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=96,
                out_channels=160,
                stride=2,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        # ------------------------------------------------------------------------------------------
        # Multi-scale contextual bottleneck
        # ------------------------------------------------------------------------------------------

        self.context_bottleneck = R3LightweightMultiScaleContext(
            in_channels=160,
            branch_channels=64,
            out_channels=192,
        )


        # ------------------------------------------------------------------------------------------
        # Lightweight decoder
        # ------------------------------------------------------------------------------------------

        self.decoder_stage2 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=(
                    192
                    +
                    96
                ),
                out_channels=128,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        self.decoder_stage1 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=(
                    128
                    +
                    64
                ),
                out_channels=96,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        self.decoder_stage0 = nn.Sequential(

            R3DepthwiseSeparableBlock(
                in_channels=(
                    96
                    +
                    32
                ),
                out_channels=64,
            ),

            R3EfficientChannelAttention(
                kernel_size=3
            ),
        )


        # ------------------------------------------------------------------------------------------
        # Lesion segmentation head
        # ------------------------------------------------------------------------------------------

        self.segmentation_head = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
        )


        # ------------------------------------------------------------------------------------------
        # Boundary prediction head
        #
        # Training supervision will later be constructed from lesion-mask boundaries.
        # At inference this remains a direct network prediction.
        # ------------------------------------------------------------------------------------------

        self.boundary_head = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
        )


        # ------------------------------------------------------------------------------------------
        # Lesion-aware diagnostic head
        #
        # Diagnostic evidence =
        #
        #   [global bottleneck pooling || predicted-lesion-weighted bottleneck pooling]
        #
        # 192 + 192 = 384 features.
        # ------------------------------------------------------------------------------------------

        self.diagnostic_head = nn.Sequential(

            nn.Linear(
                in_features=384,
                out_features=96,
            ),

            nn.SiLU(),

            nn.Dropout(
                p=0.20
            ),

            nn.Linear(
                in_features=96,
                out_features=num_diagnostic_classes,
            ),
        )


    def forward(
        self,
        x
    ):

        # ------------------------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------------------------

        stem = self.stem(
            x
        )


        encoder_1 = self.encoder_stage1(
            stem
        )


        encoder_2 = self.encoder_stage2(
            encoder_1
        )


        encoder_3 = self.encoder_stage3(
            encoder_2
        )


        bottleneck = self.context_bottleneck(
            encoder_3
        )


        # ------------------------------------------------------------------------------------------
        # Decoder stage 2
        # ------------------------------------------------------------------------------------------

        up_2 = F.interpolate(
            bottleneck,
            size=encoder_2.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        decoder_2 = self.decoder_stage2(

            torch.cat(
                [
                    up_2,
                    encoder_2,
                ],
                dim=1,
            )
        )


        # ------------------------------------------------------------------------------------------
        # Decoder stage 1
        # ------------------------------------------------------------------------------------------

        up_1 = F.interpolate(
            decoder_2,
            size=encoder_1.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        decoder_1 = self.decoder_stage1(

            torch.cat(
                [
                    up_1,
                    encoder_1,
                ],
                dim=1,
            )
        )


        # ------------------------------------------------------------------------------------------
        # Full-resolution decoder
        # ------------------------------------------------------------------------------------------

        up_0 = F.interpolate(
            decoder_1,
            size=stem.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        full_resolution_features = self.decoder_stage0(

            torch.cat(
                [
                    up_0,
                    stem,
                ],
                dim=1,
            )
        )


        # ------------------------------------------------------------------------------------------
        # Segmentation + boundary outputs
        # ------------------------------------------------------------------------------------------

        segmentation_logits = self.segmentation_head(
            full_resolution_features
        )


        boundary_logits = self.boundary_head(
            full_resolution_features
        )


        # ------------------------------------------------------------------------------------------
        # Predicted lesion attention
        #
        # NO ground-truth mask enters this pathway.
        # ------------------------------------------------------------------------------------------

        predicted_probability = torch.sigmoid(
            segmentation_logits
        )


        lesion_attention = F.interpolate(
            predicted_probability,
            size=bottleneck.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        # ------------------------------------------------------------------------------------------
        # Global diagnostic evidence
        # ------------------------------------------------------------------------------------------

        global_context = (
            F.adaptive_avg_pool2d(
                bottleneck,
                output_size=1,
            )
            .flatten(
                1
            )
        )


        # ------------------------------------------------------------------------------------------
        # Predicted-lesion-weighted diagnostic evidence
        # ------------------------------------------------------------------------------------------

        weighted_feature_sum = (
            bottleneck
            *
            lesion_attention
        ).sum(
            dim=(
                2,
                3
            )
        )


        attention_mass = lesion_attention.sum(
            dim=(
                2,
                3
            )
        )


        lesion_context = (
            weighted_feature_sum
            /
            (
                attention_mass
                +
                1e-6
            )
        )


        diagnostic_features = torch.cat(
            [
                global_context,
                lesion_context,
            ],
            dim=1,
        )


        diagnostic_logits = self.diagnostic_head(
            diagnostic_features
        )


        return {

            "logits":
                segmentation_logits,

            "boundary_logits":
                boundary_logits,

            "diagnostic_logits":
                diagnostic_logits,

            "features":
                full_resolution_features,

            "bottleneck":
                bottleneck,

            "lesion_attention":
                lesion_attention,
        }


def create_lwnnet_bcd_reviewer_upgrade_model(
    seed=20260830,
    device=None,
    num_diagnostic_classes=3,
):

    reset_r3_seed(
        seed
    )


    model = LWNNetBCDReviewerUpgradeCandidate(
        num_diagnostic_classes=num_diagnostic_classes
    )


    if device is not None:

        model = model.to(
            device
        )


    return model


def reset_r3_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


def reset_all_seeds(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


class R4DiagnosticFeatureAdapter(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.projection = nn.Conv2d(
            in_channels=192,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )


        self.normalization = nn.GroupNorm(
            num_groups=8,
            num_channels=96,
        )


        self.activation = nn.SiLU(
            inplace=True
        )


    def forward(
        self,
        x
    ):

        x = self.projection(
            x
        )


        x = self.normalization(
            x
        )


        x = self.activation(
            x
        )


        return x


class R4EvidenceGate(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.fc1 = nn.Linear(
            in_features=288,
            out_features=48,
            bias=True,
        )


        self.activation = nn.SiLU(
            inplace=True
        )


        self.fc2 = nn.Linear(
            in_features=48,
            out_features=3,
            bias=True,
        )


    def forward(
        self,
        global_descriptor,
        lesion_descriptor,
        boundary_descriptor,
    ):

        concatenated = torch.cat(
            [
                global_descriptor,
                lesion_descriptor,
                boundary_descriptor,
            ],
            dim=1,
        )


        gate_logits = self.fc1(
            concatenated
        )


        gate_logits = self.activation(
            gate_logits
        )


        gate_logits = self.fc2(
            gate_logits
        )


        # ------------------------------------------------------------------------------------------
        # FP32 softmax is deliberate for numerical stability under future AMP execution.
        # ------------------------------------------------------------------------------------------

        gate_weights_fp32 = torch.softmax(
            gate_logits.float(),
            dim=1,
        )


        gate_weights = gate_weights_fp32.to(
            dtype=global_descriptor.dtype
        )


        descriptors = torch.stack(
            [
                global_descriptor,
                lesion_descriptor,
                boundary_descriptor,
            ],
            dim=1,
        )


        fused_descriptor = (
            descriptors
            *
            gate_weights.unsqueeze(
                -1
            )
        ).sum(
            dim=1
        )


        return (
            fused_descriptor,
            gate_weights_fp32,
            gate_logits,
        )


class R4HierarchicalBinaryHead(
    nn.Module
):

    def __init__(
        self,
        dropout_probability=0.20,
    ):

        super().__init__()


        self.fc1 = nn.Linear(
            in_features=96,
            out_features=32,
            bias=True,
        )


        self.activation = nn.SiLU(
            inplace=True
        )


        self.dropout = nn.Dropout(
            p=float(
                dropout_probability
            )
        )


        self.fc2 = nn.Linear(
            in_features=32,
            out_features=1,
            bias=True,
        )


    def forward(
        self,
        x
    ):

        x = self.fc1(
            x
        )


        x = self.activation(
            x
        )


        x = self.dropout(
            x
        )


        x = self.fc2(
            x
        )


        return x


class R4HierarchicalLesionAwareFeatureMappingLWNNetBCD(
    nn.Module
):

    def __init__(
        self,
        frozen_r3_structure
    ):

        super().__init__()


        REQUIRED_R3_ATTRIBUTES = [
            "stem",
            "encoder_stage1",
            "encoder_stage2",
            "encoder_stage3",
            "context_bottleneck",
            "decoder_stage2",
            "decoder_stage1",
            "decoder_stage0",
            "segmentation_head",
            "boundary_head",
            "diagnostic_head",
        ]


        missing_attributes = [
            attribute
            for attribute in REQUIRED_R3_ATTRIBUTES
            if not hasattr(
                frozen_r3_structure,
                attribute
            )
        ]


        if missing_attributes:

            raise RuntimeError(
                "R4-00 STOPPED: frozen R3 architecture is missing required modules:\n"
                f"{missing_attributes}"
            )


        # ==========================================================================================
        # RETAIN EXACT R3 LIGHTWEIGHT LOCALIZATION BACKBONE
        #
        # IMPORTANT:
        # We intentionally DO NOT store frozen_r3_structure itself as a child module.
        # Therefore the old generic R3 diagnostic_head is NOT registered in R4.
        # ==========================================================================================

        self.stem = frozen_r3_structure.stem


        self.encoder_stage1 = frozen_r3_structure.encoder_stage1


        self.encoder_stage2 = frozen_r3_structure.encoder_stage2


        self.encoder_stage3 = frozen_r3_structure.encoder_stage3


        self.context_bottleneck = frozen_r3_structure.context_bottleneck


        self.decoder_stage2 = frozen_r3_structure.decoder_stage2


        self.decoder_stage1 = frozen_r3_structure.decoder_stage1


        self.decoder_stage0 = frozen_r3_structure.decoder_stage0


        self.segmentation_head = frozen_r3_structure.segmentation_head


        self.boundary_head = frozen_r3_structure.boundary_head


        # ==========================================================================================
        # R4 TITLE-ALIGNED DIAGNOSTIC PATHWAY
        # ==========================================================================================

        self.diagnostic_feature_adapter = R4DiagnosticFeatureAdapter()


        self.evidence_gate = R4EvidenceGate()


        self.lesion_presence_head = R4HierarchicalBinaryHead(
            dropout_probability=0.20
        )


        self.malignancy_given_lesion_head = R4HierarchicalBinaryHead(
            dropout_probability=0.20
        )


    @staticmethod
    def weighted_feature_pool(
        feature_map,
        attention_map,
        epsilon=1e-6,
    ):

        if feature_map.shape[
            -2:
        ] != attention_map.shape[
            -2:
        ]:

            raise RuntimeError(
                "R4 weighted feature pooling received spatially mismatched tensors."
            )


        if attention_map.shape[
            1
        ] != 1:

            raise RuntimeError(
                "R4 weighted feature pooling expects one-channel attention."
            )


        weighted_sum = (
            feature_map
            *
            attention_map
        ).sum(
            dim=(
                2,
                3
            )
        )


        attention_mass = attention_map.sum(
            dim=(
                2,
                3
            )
        )


        descriptor = (
            weighted_sum
            /
            (
                attention_mass
                +
                float(
                    epsilon
                )
            )
        )


        return descriptor


    def forward(
        self,
        x
    ):

        # ==========================================================================================
        # SHARED LIGHTWEIGHT ENCODER
        # ==========================================================================================

        stem = self.stem(
            x
        )


        encoder_1 = self.encoder_stage1(
            stem
        )


        encoder_2 = self.encoder_stage2(
            encoder_1
        )


        encoder_3 = self.encoder_stage3(
            encoder_2
        )


        bottleneck = self.context_bottleneck(
            encoder_3
        )


        # ==========================================================================================
        # R3 LOCALIZATION DECODER
        # ==========================================================================================

        up_2 = F.interpolate(
            bottleneck,
            size=encoder_2.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        decoder_2 = self.decoder_stage2(
            torch.cat(
                [
                    up_2,
                    encoder_2,
                ],
                dim=1,
            )
        )


        up_1 = F.interpolate(
            decoder_2,
            size=encoder_1.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        decoder_1 = self.decoder_stage1(
            torch.cat(
                [
                    up_1,
                    encoder_1,
                ],
                dim=1,
            )
        )


        up_0 = F.interpolate(
            decoder_1,
            size=stem.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        full_resolution_features = self.decoder_stage0(
            torch.cat(
                [
                    up_0,
                    stem,
                ],
                dim=1,
            )
        )


        segmentation_logits = self.segmentation_head(
            full_resolution_features
        )


        boundary_logits = self.boundary_head(
            full_resolution_features
        )


        # ==========================================================================================
        # TASK-SPECIFIC DIAGNOSTIC FEATURE ADAPTER
        # ==========================================================================================

        diagnostic_feature_map = self.diagnostic_feature_adapter(
            bottleneck
        )


        # ==========================================================================================
        # ASYMMETRIC LESION / BOUNDARY CONDITIONING
        #
        # CRITICAL:
        # The predicted localization maps are DETACHED before diagnostic conditioning.
        #
        # Diagnostic loss can use localization evidence but cannot directly alter segmentation
        # or boundary heads through these attention maps.
        # ==========================================================================================

        segmentation_probability_detached = (
            torch.sigmoid(
                segmentation_logits.float()
            )
            .detach()
        )


        boundary_probability_detached = (
            torch.sigmoid(
                boundary_logits.float()
            )
            .detach()
        )


        lesion_attention = F.interpolate(
            segmentation_probability_detached,
            size=diagnostic_feature_map.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        boundary_attention = F.interpolate(
            boundary_probability_detached,
            size=diagnostic_feature_map.shape[
                -2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        lesion_attention = lesion_attention.to(
            dtype=diagnostic_feature_map.dtype
        )


        boundary_attention = boundary_attention.to(
            dtype=diagnostic_feature_map.dtype
        )


        # ==========================================================================================
        # EXPLICIT FEATURE MAPPING
        # ==========================================================================================

        global_descriptor = (
            F.adaptive_avg_pool2d(
                diagnostic_feature_map,
                output_size=1,
            )
            .flatten(
                1
            )
        )


        lesion_descriptor = self.weighted_feature_pool(
            feature_map=diagnostic_feature_map,
            attention_map=lesion_attention,
        )


        boundary_descriptor = self.weighted_feature_pool(
            feature_map=diagnostic_feature_map,
            attention_map=boundary_attention,
        )


        # ==========================================================================================
        # EXPLAINABLE EVIDENCE FUSION
        # ==========================================================================================

        (
            fused_descriptor,
            evidence_gate_weights,
            evidence_gate_logits,
        ) = self.evidence_gate(
            global_descriptor=global_descriptor,
            lesion_descriptor=lesion_descriptor,
            boundary_descriptor=boundary_descriptor,
        )


        # ==========================================================================================
        # HIERARCHICAL BREAST-CANCER DETECTOR
        # ==========================================================================================

        lesion_presence_logit = self.lesion_presence_head(
            fused_descriptor
        )


        malignancy_given_lesion_logit = self.malignancy_given_lesion_head(
            fused_descriptor
        )


        # ------------------------------------------------------------------------------------------
        # Compute clinically interpretable probabilities in FP32.
        # ------------------------------------------------------------------------------------------

        lesion_presence_probability = torch.sigmoid(
            lesion_presence_logit.float()
        )


        malignancy_given_lesion_probability = torch.sigmoid(
            malignancy_given_lesion_logit.float()
        )


        normal_probability = (
            1.0
            -
            lesion_presence_probability
        )


        benign_probability = (
            lesion_presence_probability
            *
            (
                1.0
                -
                malignancy_given_lesion_probability
            )
        )


        malignant_probability = (
            lesion_presence_probability
            *
            malignancy_given_lesion_probability
        )


        diagnostic_class_probabilities = torch.cat(
            [
                normal_probability,
                benign_probability,
                malignant_probability,
            ],
            dim=1,
        )


        cancer_probability = malignant_probability


        return {

            # --------------------------------------------------------------------------------------
            # Localization
            # --------------------------------------------------------------------------------------

            "logits":
                segmentation_logits,

            "segmentation_logits":
                segmentation_logits,

            "boundary_logits":
                boundary_logits,

            # --------------------------------------------------------------------------------------
            # Hierarchical detection
            # --------------------------------------------------------------------------------------

            "lesion_presence_logit":
                lesion_presence_logit,

            "lesion_presence_probability":
                lesion_presence_probability,

            "malignancy_given_lesion_logit":
                malignancy_given_lesion_logit,

            "malignancy_given_lesion_probability":
                malignancy_given_lesion_probability,

            "diagnostic_class_probs":
                diagnostic_class_probabilities,

            "cancer_probability":
                cancer_probability,

            # --------------------------------------------------------------------------------------
            # Explicit feature mapping / explainability
            # --------------------------------------------------------------------------------------

            "global_descriptor":
                global_descriptor,

            "lesion_descriptor":
                lesion_descriptor,

            "boundary_descriptor":
                boundary_descriptor,

            "fused_descriptor":
                fused_descriptor,

            "evidence_gate_weights":
                evidence_gate_weights,

            "evidence_gate_logits":
                evidence_gate_logits,

            "lesion_attention":
                lesion_attention,

            "boundary_attention":
                boundary_attention,

            # --------------------------------------------------------------------------------------
            # Internal feature outputs
            # --------------------------------------------------------------------------------------

            "features":
                full_resolution_features,

            "bottleneck":
                bottleneck,

            "diagnostic_feature_map":
                diagnostic_feature_map,
        }


def create_r4_title_aligned_lwnnet_bcd(
    seed=20260830,
    device=None,
):

    reset_all_seeds(
        seed
    )


    frozen_r3_structure = create_lwnnet_bcd_reviewer_upgrade_model(
        seed=seed,
        device=None,
        num_diagnostic_classes=3,
    )


    model = R4HierarchicalLesionAwareFeatureMappingLWNNetBCD(
        frozen_r3_structure=frozen_r3_structure
    )


    if device is not None:

        model = model.to(
            device
        )


    return model
