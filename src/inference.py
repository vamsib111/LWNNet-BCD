"""
LWNNet-BCD public inference interface.

This module is a controlled public adaptation of the authenticated frozen
R4 inference implementation.

Historical inference authority:
    SHA256
    2ec696f5341aab3a0525ae53549bfc5824bba52d156e9bb564715b1dc92cd6b6

Diagnostic semantic-repair authority:
    SHA256
    ebc9fc0f94f191026a5b92238efdb2a23050747e295e09d4dbeaa9641e1540ab

Important output semantics
--------------------------
1. ``diagnostic_class_probs`` already contains normalized probabilities.
   No additional softmax is applied.

2. Class order is:
       Normal, Benign, Malignant

3. The malignant-versus-rest cancer score is P(Malignant).

4. ``segmentation_logits`` are converted to probabilities with exactly
   one sigmoid operation.

5. The authenticated primary inference kernel did not convert
   ``boundary_logits`` into a boundary-probability map. Therefore this
   public interface preserves and returns native boundary logits.

The public prediction interface accepts an in-memory RGB NumPy image.
Ground-truth masks or diagnostic labels are not required for inference.
"""

from pathlib import Path
import hashlib

import numpy as np
import torch

from .model import create_r4_title_aligned_lwnnet_bcd
from .preprocessing import apply_exact_r4_preprocessing


MASTER_SEED = 20260830

IMAGE_SIZE = 128

EXPECTED_REGISTERED_PARAMETERS = 210_684

CLASS_ORDER = (
    "Normal",
    "Benign",
    "Malignant",
)

MALIGNANT_INDEX = 2

CANCER_THRESHOLD = 0.5

SEGMENTATION_THRESHOLD = 0.5

DIAGNOSTIC_FIELD = "diagnostic_class_probs"

SEGMENTATION_FIELD = "segmentation_logits"

BOUNDARY_FIELD = "boundary_logits"

MODEL_CANCER_FIELD = "cancer_probability"

EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "ae7371c02e765913757cf0def39c45cf41f6804541149a7ddb374fb016401560"
)


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    """Return SHA256 of a file."""
    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _load_plain_state_dict(
    checkpoint_path,
):
    """
    Load the frozen plain PyTorch state_dict container.

    The reported final LWNNet-BCD checkpoint is a plain state_dict containing
    86 tensors.
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    try:

        checkpoint_object = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:

        checkpoint_object = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint_object,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a state_dict mapping."
        )

    if not checkpoint_object:

        raise RuntimeError(
            "Checkpoint state_dict is empty."
        )

    if not all(
        isinstance(
            key,
            str,
        )
        for key in checkpoint_object.keys()
    ):

        raise RuntimeError(
            "Checkpoint state_dict contains a non-string key."
        )

    if not all(
        torch.is_tensor(
            value
        )
        for value in checkpoint_object.values()
    ):

        raise RuntimeError(
            "Checkpoint is not the expected plain tensor state_dict."
        )

    return checkpoint_object


def load_lwnnet_bcd(
    checkpoint_path,
    device=None,
    expected_sha256=None,
    seed=MASTER_SEED,
):
    """
    Load LWNNet-BCD using strict state_dict binding.

    Parameters
    ----------
    checkpoint_path:
        Path to a plain LWNNet-BCD state_dict checkpoint.

    device:
        PyTorch device. If omitted, CUDA is selected when available,
        otherwise CPU is used.

    expected_sha256:
        Optional checkpoint SHA256. Use
        ``EXPECTED_FINAL_CHECKPOINT_SHA256`` to authenticate the reported
        frozen final model.

    seed:
        Architecture initialization seed before state_dict loading.
        The reported final model used 20260830.
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if expected_sha256 is not None:

        expected_sha256 = str(
            expected_sha256
        ).lower()

        observed_sha256 = sha256_file(
            checkpoint_path
        )

        if observed_sha256.lower() != expected_sha256:

            raise RuntimeError(
                "Checkpoint SHA256 mismatch. "
                f"Observed={observed_sha256}; "
                f"Expected={expected_sha256}"
            )

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device
        )

    state_dict = _load_plain_state_dict(
        checkpoint_path
    )

    model = create_r4_title_aligned_lwnnet_bcd(
        seed=int(
            seed
        ),
        device="cpu",
    )

    if not isinstance(
        model,
        torch.nn.Module,
    ):

        raise RuntimeError(
            "LWNNet-BCD factory did not return torch.nn.Module."
        )

    registered_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if (
        registered_parameter_count
        !=
        EXPECTED_REGISTERED_PARAMETERS
    ):

        raise RuntimeError(
            "Unexpected LWNNet-BCD parameter count. "
            f"Observed={registered_parameter_count:,}; "
            f"Expected={EXPECTED_REGISTERED_PARAMETERS:,}"
        )

    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        load_result.missing_keys
        or
        load_result.unexpected_keys
    ):

        raise RuntimeError(
            "Strict state_dict binding returned incompatible keys."
        )

    model = model.to(
        device
    )

    model.eval()

    return model


def prepare_rgb_tensor(
    raw_image_rgb,
    device,
):
    """
    Apply the authenticated R4 preprocessing pipeline to an in-memory RGB image.
    """
    if not isinstance(
        raw_image_rgb,
        np.ndarray,
    ):

        raise TypeError(
            "raw_image_rgb must be a NumPy array."
        )

    if (
        raw_image_rgb.ndim != 3
        or
        raw_image_rgb.shape[2] != 3
    ):

        raise ValueError(
            "raw_image_rgb must have shape [H,W,3]."
        )

    processed = apply_exact_r4_preprocessing(
        raw_image_rgb.copy()
    )

    if not isinstance(
        processed,
        np.ndarray,
    ):

        raise RuntimeError(
            "Authenticated preprocessing did not return NumPy array."
        )

    if processed.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    ):

        raise RuntimeError(
            "Unexpected preprocessed image shape: "
            f"{processed.shape}"
        )

    if processed.dtype != np.float32:

        processed = processed.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(
        processed
    ).all():

        raise RuntimeError(
            "Preprocessed image contains non-finite values."
        )

    minimum = float(
        processed.min()
    )

    maximum = float(
        processed.max()
    )

    if minimum < 0.0 or maximum > 1.0:

        raise RuntimeError(
            "Preprocessed image is outside expected [0,1] range."
        )

    chw = np.ascontiguousarray(
        processed.transpose(
            2,
            0,
            1,
        )
    )

    tensor = torch.from_numpy(
        chw
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        device=torch.device(
            device
        ),
        dtype=torch.float32,
    )

    return tensor


def predict_tensor_batch(
    model,
    image_tensor,
    segmentation_threshold=SEGMENTATION_THRESHOLD,
):
    """
    Run image-only inference on a preprocessed tensor batch.

    Expected tensor shape:
        [B,3,128,128]

    Boundary output is intentionally returned as native logits because the
    authenticated primary inference kernel did not apply a boundary sigmoid.
    """
    if not isinstance(
        model,
        torch.nn.Module,
    ):

        raise TypeError(
            "model must be torch.nn.Module."
        )

    if not torch.is_tensor(
        image_tensor
    ):

        raise TypeError(
            "image_tensor must be a PyTorch tensor."
        )

    if (
        image_tensor.ndim != 4
        or
        image_tensor.shape[1] != 3
        or
        tuple(
            image_tensor.shape[-2:]
        )
        !=
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ):

        raise ValueError(
            "image_tensor must have shape [B,3,128,128]."
        )

    threshold = float(
        segmentation_threshold
    )

    if not (
        0.0
        <=
        threshold
        <=
        1.0
    ):

        raise ValueError(
            "segmentation_threshold must be within [0,1]."
        )

    try:

        model_device = next(
            model.parameters()
        ).device

    except StopIteration:

        raise RuntimeError(
            "Model contains no parameters."
        )

    if image_tensor.device != model_device:

        image_tensor = image_tensor.to(
            model_device
        )

    model.eval()

    with torch.inference_mode():

        output = model(
            image_tensor
        )

        if not isinstance(
            output,
            dict,
        ):

            raise RuntimeError(
                "LWNNet-BCD forward output must be a dictionary."
            )

        required_fields = {
            DIAGNOSTIC_FIELD,
            SEGMENTATION_FIELD,
            BOUNDARY_FIELD,
        }

        missing_fields = (
            required_fields
            -
            set(
                output.keys()
            )
        )

        if missing_fields:

            raise RuntimeError(
                "Missing required inference outputs: "
                f"{sorted(missing_fields)}"
            )

        native_probabilities = output[
            DIAGNOSTIC_FIELD
        ].float()

        segmentation_logits = output[
            SEGMENTATION_FIELD
        ].float()

        boundary_logits = output[
            BOUNDARY_FIELD
        ].float()

        batch_size = int(
            image_tensor.shape[
                0
            ]
        )

        if tuple(
            native_probabilities.shape
        ) != (
            batch_size,
            3,
        ):

            raise RuntimeError(
                "Unexpected diagnostic probability shape: "
                f"{tuple(native_probabilities.shape)}"
            )

        if tuple(
            segmentation_logits.shape
        ) != (
            batch_size,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            raise RuntimeError(
                "Unexpected segmentation-logit shape: "
                f"{tuple(segmentation_logits.shape)}"
            )

        if tuple(
            boundary_logits.shape
        ) != (
            batch_size,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            raise RuntimeError(
                "Unexpected boundary-logit shape: "
                f"{tuple(boundary_logits.shape)}"
            )

        for name, tensor in [
            (
                "diagnostic_class_probs",
                native_probabilities,
            ),
            (
                "segmentation_logits",
                segmentation_logits,
            ),
            (
                "boundary_logits",
                boundary_logits,
            ),
        ]:

            if not torch.isfinite(
                tensor
            ).all():

                raise RuntimeError(
                    f"Non-finite values detected in {name}."
                )

        if bool(
            (
                native_probabilities
                <
                -1e-6
            ).any()
        ):

            raise RuntimeError(
                "diagnostic_class_probs contains values below zero."
            )

        if bool(
            (
                native_probabilities
                >
                1.0 + 1e-6
            ).any()
        ):

            raise RuntimeError(
                "diagnostic_class_probs contains values above one."
            )

        probability_sums = native_probabilities.sum(
            dim=1
        )

        if not torch.allclose(
            probability_sums,
            torch.ones_like(
                probability_sums
            ),
            rtol=1e-5,
            atol=1e-6,
        ):

            raise RuntimeError(
                "diagnostic_class_probs does not sum to one."
            )

        # AUTHENTICATED SEMANTIC RULE:
        # diagnostic_class_probs is already normalized.
        # DO NOT apply softmax here.

        predicted_class_indices = torch.argmax(
            native_probabilities,
            dim=1,
        )

        cancer_probabilities = native_probabilities[
            :,
            MALIGNANT_INDEX,
        ]

        cancer_predictions = (
            cancer_probabilities
            >=
            CANCER_THRESHOLD
        )

        # AUTHENTICATED LOCALIZATION RULE:
        # segmentation_logits -> sigmoid exactly once.
        segmentation_probabilities = torch.sigmoid(
            segmentation_logits
        )

        segmentation_masks = (
            segmentation_probabilities
            >=
            threshold
        )

        # If the model exposes its native cancer_probability field,
        # verify the hierarchical identity but do not substitute a
        # different score definition.
        if MODEL_CANCER_FIELD in output:

            model_cancer_probability = output[
                MODEL_CANCER_FIELD
            ].float().reshape(
                -1
            )

            if tuple(
                model_cancer_probability.shape
            ) != (
                batch_size,
            ):

                raise RuntimeError(
                    "Unexpected model cancer_probability shape."
                )

            if not torch.allclose(
                model_cancer_probability,
                cancer_probabilities,
                rtol=1e-6,
                atol=1e-7,
            ):

                raise RuntimeError(
                    "Model cancer_probability is not identical to "
                    "P(Malignant)."
                )

        return {
            "diagnostic_class_probs":
                native_probabilities.detach().cpu(),

            "predicted_class_indices":
                predicted_class_indices.detach().cpu(),

            "cancer_probabilities":
                cancer_probabilities.detach().cpu(),

            "cancer_predictions":
                cancer_predictions.detach().cpu(),

            "segmentation_logits":
                segmentation_logits.detach().cpu(),

            "segmentation_probabilities":
                segmentation_probabilities.detach().cpu(),

            "segmentation_masks":
                segmentation_masks.detach().cpu(),

            # Intentionally native logits.
            "boundary_logits":
                boundary_logits.detach().cpu(),
        }


def predict_rgb(
    model,
    raw_image_rgb,
    segmentation_threshold=SEGMENTATION_THRESHOLD,
):
    """
    Run LWNNet-BCD inference on one in-memory RGB image.

    Returns a user-facing prediction dictionary. No ground-truth information
    is needed.
    """
    try:

        device = next(
            model.parameters()
        ).device

    except StopIteration:

        raise RuntimeError(
            "Model contains no parameters."
        )

    tensor = prepare_rgb_tensor(
        raw_image_rgb,
        device=device,
    )

    batch_output = predict_tensor_batch(
        model=model,
        image_tensor=tensor,
        segmentation_threshold=segmentation_threshold,
    )

    class_index = int(
        batch_output[
            "predicted_class_indices"
        ][
            0
        ].item()
    )

    diagnostic_probabilities = (
        batch_output[
            "diagnostic_class_probs"
        ][
            0
        ].numpy()
    )

    cancer_probability = float(
        batch_output[
            "cancer_probabilities"
        ][
            0
        ].item()
    )

    return {
        "class_order":
            CLASS_ORDER,

        "diagnostic_probabilities":
            diagnostic_probabilities,

        "predicted_class_index":
            class_index,

        "predicted_class":
            CLASS_ORDER[
                class_index
            ],

        "cancer_probability":
            cancer_probability,

        "cancer_prediction":
            bool(
                batch_output[
                    "cancer_predictions"
                ][
                    0
                ].item()
            ),

        "cancer_threshold":
            CANCER_THRESHOLD,

        "segmentation_probability":
            batch_output[
                "segmentation_probabilities"
            ][
                0,
                0,
            ].numpy(),

        "segmentation_mask":
            batch_output[
                "segmentation_masks"
            ][
                0,
                0,
            ].numpy().astype(
                np.uint8
            ),

        "segmentation_threshold":
            float(
                segmentation_threshold
            ),

        # Native boundary output only.
        "boundary_logits":
            batch_output[
                "boundary_logits"
            ][
                0,
                0,
            ].numpy(),
    }


__all__ = [
    "MASTER_SEED",
    "IMAGE_SIZE",
    "EXPECTED_REGISTERED_PARAMETERS",
    "CLASS_ORDER",
    "MALIGNANT_INDEX",
    "CANCER_THRESHOLD",
    "SEGMENTATION_THRESHOLD",
    "EXPECTED_FINAL_CHECKPOINT_SHA256",
    "sha256_file",
    "load_lwnnet_bcd",
    "prepare_rgb_tensor",
    "predict_tensor_batch",
    "predict_rgb",
]
