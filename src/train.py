"""
LWNNet-BCD final-development training implementation.

This module is a controlled public adaptation of the authenticated historical
R4-08C-B-R1 final-training source. The reported final model used 614
development images, deterministic seed 20260830, 26 fixed epochs, AdamW,
automatic mixed precision, and the frozen joint objective.

The historical implementation contained additional experiment-governance,
backup, provenance, and machine-local path checks. Those controls are not
duplicated here. The optimization, batch preparation, objective, boundary
construction, deterministic DataLoader configuration, learning-rate schedule,
and GradScaler overflow-update semantics are preserved.

Loss weights are supplied explicitly by the caller. For reproduction of the
reported final model, use the frozen full-development weights documented in
configs/lwnnet_bcd_config.yaml.
"""

import os

# Required for deterministic CUDA matrix operations when supported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import math
import random
import time

import numpy
import torch
from torch.utils.data import DataLoader

from .model import create_r4_title_aligned_lwnnet_bcd
from .losses import R4JointObjective
from .dataset import R4DevelopmentDataset
from .boundary import create_boundary_target


MASTER_SEED = 20260830
IMAGE_SIZE = 128
BATCH_SIZE = 16
FINAL_EPOCHS = 26
EXPECTED_REGISTERED_PARAMETERS = 210_684

INITIAL_LR = 0.001
LATE_LR = 0.0005
WEIGHT_DECAY = 0.0001

AMP_ENABLED = True
BOUNDARY_THICKENING_KERNEL = 3

LAMBDA_SEGMENTATION = 1.0
LAMBDA_BOUNDARY = 0.25
LAMBDA_DIAGNOSIS = 0.5

REQUIRED_LOSS_WEIGHT_KEYS = (
    "segmentation_positive_weight",
    "boundary_positive_weight",
    "presence_negative_weight",
    "presence_positive_weight",
    "malignancy_negative_weight",
    "malignancy_positive_weight",
)


def set_reproducibility(
    seed,
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

    torch.use_deterministic_algorithms(
        True
    )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def flatten_scalar_values(
    obj,
    prefix="$",
    output=None,
):

    if output is None:

        output = {}


    if torch.is_tensor(
        obj
    ):

        if obj.numel() == 1:

            value = float(
                obj.detach().cpu().item()
            )

            if math.isfinite(
                value
            ):

                output[
                    prefix
                ] = value

        return output


    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            flatten_scalar_values(
                value,
                f"{prefix}.{key}",
                output,
            )

        return output


    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):

        for index, value in enumerate(
            obj
        ):

            flatten_scalar_values(
                value,
                f"{prefix}[{index}]",
                output,
            )

        return output


    if isinstance(
        obj,
        (
            int,
            float,
            np.number,
        ),
    ):

        value = float(
            obj
        )

        if math.isfinite(
            value
        ):

            output[
                prefix
            ] = value


    return output


def locate_total_loss(
    loss_output,
):

    if (
        torch.is_tensor(
            loss_output
        )
        and
        loss_output.numel() == 1
    ):

        return (
            loss_output,
            "$",
        )


    preferred_names = {
        "total_loss",
        "loss_total",
        "joint_loss",
        "overall_loss",
        "total",
        "loss",
    }


    matches = []


    def visit(
        obj,
        path="$",
        key_name=None,
    ):

        if torch.is_tensor(
            obj
        ):

            if (
                obj.numel() == 1
                and
                key_name is not None
                and
                str(
                    key_name
                ).strip().lower()
                in
                preferred_names
            ):

                matches.append(
                    (
                        obj,
                        path,
                    )
                )

            return


        if isinstance(
            obj,
            dict,
        ):

            for key, value in obj.items():

                visit(
                    value,
                    f"{path}.{key}",
                    key,
                )

            return


        if isinstance(
            obj,
            (
                list,
                tuple,
            ),
        ):

            if (
                len(
                    obj
                )
                > 0
                and
                torch.is_tensor(
                    obj[
                        0
                    ]
                )
                and
                obj[
                    0
                ].numel()
                == 1
            ):

                matches.append(
                    (
                        obj[
                            0
                        ],
                        f"{path}[0]",
                    )
                )

                return


            for index, value in enumerate(
                obj
            ):

                visit(
                    value,
                    f"{path}[{index}]",
                    None,
                )


    visit(
        loss_output
    )


    unique = {
        path:
            tensor
        for tensor, path in matches
    }


    if len(
        unique
    ) != 1:

        raise RuntimeError(
            "R4-08C-B-R1 STOPPED BEFORE BACKWARD: unable to bind exactly "
            "one total-loss tensor.\n"
            f"Candidate paths={list(unique.keys())}\n"
            f"Scalar paths={list(flatten_scalar_values(loss_output).keys())}"
        )


    path = next(
        iter(
            unique.keys()
        )
    )

    return (
        unique[
            path
        ],
        path,
    )


def optimizer_max_step(
    optimizer,
):

    steps = []


    for state in optimizer.state.values():

        if "step" not in state:

            continue


        value = state[
            "step"
        ]


        if torch.is_tensor(
            value
        ):

            value = int(
                value.detach().cpu().item()
            )

        else:

            value = int(
                value
            )


        steps.append(
            value
        )


    if not steps:

        return 0


    return max(
        steps
    )


def verify_model_parameters_finite(
    model,
    epoch,
    batch_index,
):

    for parameter_name, parameter in model.named_parameters():

        if not bool(
            torch.isfinite(
                parameter.detach()
            ).all().item()
        ):

            raise RuntimeError(
                "R4-08C-B-R1 STOPPED: model parameter became non-finite.\n"
                f"Epoch={epoch}\n"
                f"Batch={batch_index}\n"
                f"Parameter={parameter_name}"
            )


def epoch_learning_rate(epoch):
    """
    Return the frozen final-training learning rate for a 1-based epoch.
    """
    epoch = int(epoch)

    if not 1 <= epoch <= FINAL_EPOCHS:
        raise ValueError(
            f"Epoch must be in [1, {FINAL_EPOCHS}], got {epoch}."
        )

    if epoch <= 23:
        return INITIAL_LR

    return LATE_LR


def validate_loss_weights(loss_weights):
    """
    Validate the six data-derived weights required by R4JointObjective.
    """
    if not isinstance(loss_weights, dict):
        raise TypeError(
            "loss_weights must be a dictionary."
        )

    missing = [
        key
        for key in REQUIRED_LOSS_WEIGHT_KEYS
        if key not in loss_weights
    ]

    if missing:
        raise KeyError(
            f"Missing loss weights: {missing}"
        )

    validated = {}

    for key in REQUIRED_LOSS_WEIGHT_KEYS:
        value = float(
            loss_weights[key]
        )

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Invalid positive loss weight {key}={value!r}."
            )

        validated[key] = value

    return validated


def build_final_training_components(
    records,
    loss_weights,
    device=None,
    seed=MASTER_SEED,
):
    """
    Construct the frozen final-development DataLoader, model, objective,
    optimizer, and AMP scaler.

    No validation or early-stopping loader is created.
    """
    seed = int(seed)

    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if device.type != "cuda":
        raise RuntimeError(
            "The reported final training protocol used CUDA with AMP."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to reproduce the reported final training protocol."
        )

    set_reproducibility(
        seed
    )

    weights = validate_loss_weights(
        loss_weights
    )

    train_dataset = R4DevelopmentDataset(
        records=records,
        augment=True,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    model = create_r4_title_aligned_lwnnet_bcd(
        seed=seed,
        device=device,
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if parameter_count != EXPECTED_REGISTERED_PARAMETERS:
        raise RuntimeError(
            "Unexpected model parameter count: "
            f"{parameter_count:,}."
        )

    criterion = R4JointObjective(
        segmentation_positive_weight=weights[
            "segmentation_positive_weight"
        ],
        boundary_positive_weight=weights[
            "boundary_positive_weight"
        ],
        presence_negative_weight=weights[
            "presence_negative_weight"
        ],
        presence_positive_weight=weights[
            "presence_positive_weight"
        ],
        malignancy_negative_weight=weights[
            "malignancy_negative_weight"
        ],
        malignancy_positive_weight=weights[
            "malignancy_positive_weight"
        ],
        lambda_segmentation=LAMBDA_SEGMENTATION,
        lambda_boundary=LAMBDA_BOUNDARY,
        lambda_diagnosis=LAMBDA_DIAGNOSIS,
    )

    if isinstance(
        criterion,
        torch.nn.Module,
    ):
        criterion = criterion.to(
            device
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
    )

    return {
        "device": device,
        "dataset": train_dataset,
        "loader": train_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scaler": scaler,
    }


def train_final_model(
    records,
    loss_weights,
    checkpoint_path,
    device=None,
    seed=MASTER_SEED,
):
    """
    Train the final LWNNet-BCD model using the frozen 26-epoch protocol.

    Parameters
    ----------
    records:
        Records accepted by R4DevelopmentDataset.
    loss_weights:
        Dictionary containing the six required full-development objective
        weights.
    checkpoint_path:
        Destination for the final model state_dict.
    device:
        CUDA device. The reported run used CUDA with AMP.
    seed:
        Reproducibility seed. The reported run used 20260830.

    Returns
    -------
    dict
        Model, training history, overflow events, optimizer-update totals,
        and checkpoint path.
    """
    components = build_final_training_components(
        records=records,
        loss_weights=loss_weights,
        device=device,
        seed=seed,
    )

    device = components[
        "device"
    ]

    train_loader = components[
        "loader"
    ]

    model = components[
        "model"
    ]

    criterion = components[
        "criterion"
    ]

    optimizer = components[
        "optimizer"
    ]

    scaler = components[
        "scaler"
    ]

    model.train()

    history = []
    overflow_events = []

    cumulative_attempted_batches = 0
    cumulative_successful_updates = 0
    cumulative_overflow_skips = 0

    bound_total_loss_path = None
    first_objective_scalar_paths = None

    for epoch in range(
        1,
        FINAL_EPOCHS + 1,
    ):
        epoch_lr = float(
            epoch_learning_rate(
                epoch
            )
        )

        for parameter_group in optimizer.param_groups:
            parameter_group[
                "lr"
            ] = epoch_lr

        epoch_start = time.perf_counter()

        torch.cuda.synchronize()

        epoch_samples = 0
        epoch_attempted_batches = 0
        epoch_successful_updates = 0
        epoch_overflow_skips = 0
        epoch_total_loss_sum = 0.0
        component_weighted_sums = {}

        scaler_start = float(
            scaler.get_scale()
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            if not isinstance(
                batch,
                dict,
            ):
                raise RuntimeError(
                    "DataLoader must return a dictionary batch."
                )

            required_batch_keys = {
                "image",
                "mask",
                "diagnostic_target",
            }

            missing_batch_keys = (
                required_batch_keys
                -
                set(
                    batch.keys()
                )
            )

            if missing_batch_keys:
                raise RuntimeError(
                    "Training-batch schema mismatch. "
                    f"Missing={sorted(missing_batch_keys)}; "
                    f"Available={sorted(batch.keys())}"
                )

            images = batch[
                "image"
            ].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            segmentation_targets = batch[
                "mask"
            ].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            class_targets = batch[
                "diagnostic_target"
            ].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            if segmentation_targets.ndim == 3:
                segmentation_targets = segmentation_targets.unsqueeze(
                    1
                )

            if (
                images.ndim != 4
                or images.shape[1] != 3
                or tuple(
                    images.shape[-2:]
                ) != (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ):
                raise RuntimeError(
                    "Image tensor shape mismatch: "
                    f"{tuple(images.shape)}"
                )

            if (
                segmentation_targets.ndim != 4
                or segmentation_targets.shape[1] != 1
                or tuple(
                    segmentation_targets.shape[-2:]
                ) != (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ):
                raise RuntimeError(
                    "Segmentation-target shape mismatch."
                )

            class_targets = class_targets.reshape(
                -1
            )

            batch_size_actual = int(
                images.shape[0]
            )

            if class_targets.numel() != batch_size_actual:
                raise RuntimeError(
                    "Diagnostic-target batch-size mismatch."
                )

            segmentation_targets = (
                segmentation_targets
                >=
                0.5
            ).float()

            boundary_targets = create_boundary_target(
                segmentation_targets,
                thickening_kernel=BOUNDARY_THICKENING_KERNEL,
            )

            boundary_targets = (
                boundary_targets
                >=
                0.5
            ).float()

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                outputs = model(
                    images
                )

                loss_output = criterion(
                    outputs,
                    segmentation_targets,
                    boundary_targets,
                    class_targets,
                )

                total_loss, total_loss_path = locate_total_loss(
                    loss_output
                )

            if bound_total_loss_path is None:
                bound_total_loss_path = total_loss_path

                first_objective_scalar_paths = sorted(
                    flatten_scalar_values(
                        loss_output
                    ).keys()
                )

            elif total_loss_path != bound_total_loss_path:
                raise RuntimeError(
                    "Objective total-loss binding changed. "
                    f"Expected={bound_total_loss_path}; "
                    f"Observed={total_loss_path}"
                )

            if (
                not torch.is_tensor(
                    total_loss
                )
                or total_loss.numel() != 1
            ):
                raise RuntimeError(
                    "Total loss must be a scalar tensor."
                )

            if not bool(
                torch.isfinite(
                    total_loss.detach()
                ).item()
            ):
                raise RuntimeError(
                    f"Non-finite forward total loss at "
                    f"epoch={epoch}, batch={batch_index}."
                )

            total_loss_value = float(
                total_loss.detach().cpu().item()
            )

            scalar_components = flatten_scalar_values(
                loss_output
            )

            optimizer_step_before = optimizer_max_step(
                optimizer
            )

            scaler_before = float(
                scaler.get_scale()
            )

            scaler.scale(
                total_loss
            ).backward()

            # Historical successful-run semantics:
            # unscale, then let GradScaler decide whether optimizer.step is safe.
            # There is deliberately no manual post-unscale gradient abort.
            scaler.unscale_(
                optimizer
            )

            scaler.step(
                optimizer
            )

            optimizer_step_after = optimizer_max_step(
                optimizer
            )

            scaler.update()

            scaler_after = float(
                scaler.get_scale()
            )

            optimizer_updated = (
                optimizer_step_after
                >
                optimizer_step_before
            )

            overflow_skipped = (
                not optimizer_updated
            )

            if overflow_skipped:
                if not (
                    scaler_after
                    <
                    scaler_before
                ):
                    raise RuntimeError(
                        "Optimizer update was skipped but "
                        "GradScaler did not reduce its scale."
                    )

                epoch_overflow_skips += 1
                cumulative_overflow_skips += 1

                overflow_events.append(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_attempt":
                            cumulative_attempted_batches + 1,
                        "total_loss":
                            total_loss_value,
                        "scaler_before":
                            scaler_before,
                        "scaler_after":
                            scaler_after,
                        "optimizer_step_before":
                            optimizer_step_before,
                        "optimizer_step_after":
                            optimizer_step_after,
                        "optimizer_update_skipped":
                            True,
                    }
                )

            else:
                epoch_successful_updates += 1
                cumulative_successful_updates += 1

                if (
                    optimizer_step_after
                    !=
                    optimizer_step_before + 1
                ):
                    raise RuntimeError(
                        "AdamW optimizer step counter did not advance by one."
                    )

            verify_model_parameters_finite(
                model,
                epoch,
                batch_index,
            )

            epoch_attempted_batches += 1
            cumulative_attempted_batches += 1

            epoch_samples += batch_size_actual

            epoch_total_loss_sum += (
                total_loss_value
                *
                batch_size_actual
            )

            for path, value in scalar_components.items():
                component_weighted_sums[
                    path
                ] = (
                    component_weighted_sums.get(
                        path,
                        0.0,
                    )
                    +
                    float(value)
                    *
                    batch_size_actual
                )

        if (
            epoch_successful_updates
            +
            epoch_overflow_skips
            !=
            epoch_attempted_batches
        ):
            raise RuntimeError(
                "Epoch optimizer-update accounting mismatch."
            )

        torch.cuda.synchronize()

        epoch_seconds = (
            time.perf_counter()
            -
            epoch_start
        )

        mean_total_loss = (
            epoch_total_loss_sum
            /
            max(
                epoch_samples,
                1,
            )
        )

        component_means = {
            key:
                value
                /
                max(
                    epoch_samples,
                    1,
                )
            for key, value in component_weighted_sums.items()
        }

        history.append(
            {
                "epoch":
                    epoch,

                "learning_rate":
                    epoch_lr,

                "samples":
                    epoch_samples,

                "attempted_batches":
                    epoch_attempted_batches,

                "successful_optimizer_updates":
                    epoch_successful_updates,

                "overflow_skipped_batches":
                    epoch_overflow_skips,

                "cumulative_attempted_batches":
                    cumulative_attempted_batches,

                "cumulative_successful_updates":
                    cumulative_successful_updates,

                "cumulative_overflow_skips":
                    cumulative_overflow_skips,

                "mean_total_loss":
                    mean_total_loss,

                "grad_scaler_start":
                    scaler_start,

                "grad_scaler_end":
                    float(
                        scaler.get_scale()
                    ),

                "epoch_seconds":
                    epoch_seconds,

                "component_means":
                    component_means,
            }
        )

    if (
        cumulative_successful_updates
        +
        cumulative_overflow_skips
        !=
        cumulative_attempted_batches
    ):
        raise RuntimeError(
            "Final optimizer-update accounting mismatch."
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_name(
        checkpoint_path.name
        + ".tmp"
    )

    if temporary_path.exists():
        raise RuntimeError(
            f"Temporary checkpoint already exists: {temporary_path}"
        )

    torch.save(
        model.state_dict(),
        temporary_path,
    )

    os.replace(
        temporary_path,
        checkpoint_path,
    )

    return {
        "model":
            model,

        "history":
            history,

        "overflow_events":
            overflow_events,

        "attempted_training_batches":
            cumulative_attempted_batches,

        "successful_optimizer_updates":
            cumulative_successful_updates,

        "amp_overflow_skipped_updates":
            cumulative_overflow_skips,

        "final_optimizer_step":
            optimizer_max_step(
                optimizer
            ),

        "objective_total_loss_path":
            bound_total_loss_path,

        "objective_scalar_paths":
            first_objective_scalar_paths,

        "checkpoint_path":
            checkpoint_path,
    }


__all__ = [
    "MASTER_SEED",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "FINAL_EPOCHS",
    "INITIAL_LR",
    "LATE_LR",
    "WEIGHT_DECAY",
    "AMP_ENABLED",
    "BOUNDARY_THICKENING_KERNEL",
    "LAMBDA_SEGMENTATION",
    "LAMBDA_BOUNDARY",
    "LAMBDA_DIAGNOSIS",
    "epoch_learning_rate",
    "validate_loss_weights",
    "build_final_training_components",
    "train_final_model",
]
