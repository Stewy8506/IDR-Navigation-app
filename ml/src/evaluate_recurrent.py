"""
evaluate_recurrent.py

Fast batched evaluation of Recurrent V2 (NO prior-speed conditioning)
on held-out Driver E.

Pipeline:
    IO-VNBD CSV
        ↓
    6-channel raw IMU
        ↓
    vehicle-frame alignment
        ↓
    16-channel spectral/physics features
        ↓
    32-sample windows
        ↓
    16-window temporal sequences
        ↓
    RecurrentSpeedFilterNet
        ↓
    speed + uncertainty
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from .dataset_spectral import (
    align_imu_to_vehicle_frame,
    compute_spectral_physics_features,
)
from .model import RecurrentSpeedFilterNet


# ---------------------------------------------------------------------
# Speed bins used throughout the project
# ---------------------------------------------------------------------

BINS = (
    (0, 10),
    (10, 30),
    (30, 50),
    (50, 70),
    (70, 90),
    (90, 140),
)


# ---------------------------------------------------------------------
# Robust column finder
# ---------------------------------------------------------------------

def find_column(columns, required_terms, optional_terms=None):
    """
    Robustly find a CSV column.

    Handles:
      - leading/trailing spaces
      - capitalization differences
      - UTF-8/latin1 mojibake
      - the IO-VNBD typo ACCELEROMETERY
    """

    optional_terms = optional_terms or []

    def normalize(text):
        text = str(text).strip().lower()

        # Common encoding artifacts
        replacements = {
            "â²": "²",
            "Â²": "²",
            "â°": "°",
            "Â°": "°",
            "Î¼": "μ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Make matching tolerant of spacing/symbols
        text = text.replace(" ", "")
        text = text.replace("_", "")
        text = text.replace("-", "")

        return text

    normalized_columns = {
        normalize(col): col
        for col in columns
    }

    # First try exact normalized substring matching
    for normalized, original in normalized_columns.items():

        if all(
            normalize(term) in normalized
            for term in required_terms
        ):
            if all(
                normalize(term) in normalized
                for term in optional_terms
            ):
                return original

    return None


# ---------------------------------------------------------------------
# Find the six raw IMU channels
# ---------------------------------------------------------------------

def find_imu_columns(df_s):
    """
    Locate the six raw IMU channels.

    Required order:

        0 = Accelerometer X
        1 = Accelerometer Y
        2 = Accelerometer Z
        3 = Gyroscope Yaw
        4 = Gyroscope Pitch
        5 = Gyroscope Roll

    IO-VNBD has a known typo:
        ACCELEROMETERY = Accelerometer Y
    """

    columns = list(df_s.columns)

    # -----------------------------
    # Accelerometer X
    # -----------------------------

    accel_x = find_column(
        columns,
        ["accelerometer", "x"],
    )

    # -----------------------------
    # Accelerometer Y
    #
    # Dataset typo:
    # ACCELEROMETERY (m/s²)
    # -----------------------------

    accel_y = find_column(
        columns,
        ["accelerometery"],
    )

    if accel_y is None:
        # Fallback in case another file uses normal spelling
        accel_y = find_column(
            columns,
            ["accelerometer", "y"],
        )

    # -----------------------------
    # Accelerometer Z
    # -----------------------------

    accel_z = find_column(
        columns,
        ["accelerometer", "z"],
    )

    # -----------------------------
    # Gyroscope Yaw
    # -----------------------------

    gyro_yaw = find_column(
        columns,
        ["gyroscope", "yaw"],
    )

    # -----------------------------
    # Gyroscope Pitch
    # -----------------------------

    gyro_pitch = find_column(
        columns,
        ["gyroscope", "pitch"],
    )

    # -----------------------------
    # Gyroscope Roll
    # -----------------------------

    gyro_roll = find_column(
        columns,
        ["gyroscope", "roll"],
    )

    found = {
        "ACCELEROMETER X": accel_x,
        "ACCELEROMETER Y": accel_y,
        "ACCELEROMETER Z": accel_z,
        "GYROSCOPE Yaw": gyro_yaw,
        "GYROSCOPE Pitch": gyro_pitch,
        "GYROSCOPE Roll": gyro_roll,
    }

    missing = [
        name
        for name, value in found.items()
        if value is None
    ]

    if missing:
        raise KeyError(
            "Could not identify required IMU columns: "
            + ", ".join(missing)
            + "\nAvailable columns:\n"
            + "\n".join(
                f"  - {repr(c)}"
                for c in columns
            )
        )

    return (
        accel_x,
        accel_y,
        accel_z,
        gyro_yaw,
        gyro_pitch,
        gyro_roll,
    )


# ---------------------------------------------------------------------
# Find vehicle ground-truth speed column
# ---------------------------------------------------------------------

def find_speed_column(df_v):

    columns = list(df_v.columns)

    candidates = [
        "Indicated Vehicle Speed (km/hr)",
        "Velocity (km/hr)",
        "GPS SPEED (Kmh)",
    ]

    # First try exact names
    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Then robust matching
    for col in columns:

        normalized = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
        )

        if (
            "indicatedvehiclespeed" in normalized
            and "km" in normalized
        ):
            return col

        if (
            "velocity" in normalized
            and "km" in normalized
        ):
            return col

        if (
            "gpsspeed" in normalized
            and "kmh" in normalized
        ):
            return col

    raise KeyError(
        "Could not find vehicle speed column.\n"
        "Available columns:\n"
        + "\n".join(
            f"  - {repr(c)}"
            for c in columns
        )
    )


# ---------------------------------------------------------------------
# Load Driver E sequences
# ---------------------------------------------------------------------

def load_driver_e_sequences(
    data_dir: str,
    window_size: int = 32,
    seq_len: int = 16,
    step_size: int = 8,
):

    s_files = sorted(
        sf
        for sf in glob.glob(
            os.path.join(
                data_dir,
                "**",
                "S-*.csv",
            ),
            recursive=True,
        )
        if "Driver E" in sf
    )

    sequences = []

    print(
        f"Loading {len(s_files)} Driver E sensor files..."
    )

    for file_idx, s_file in enumerate(
        s_files,
        1,
    ):

        v_file = os.path.join(
            os.path.dirname(s_file),
            os.path.basename(s_file).replace(
                "S-",
                "V-",
            ),
        )

        if not os.path.exists(v_file):
            print(
                f"\nWARNING: Vehicle file missing:"
                f"\n  {v_file}"
            )
            continue

        try:

            # ---------------------------------------------------------
            # Read CSV files
            # ---------------------------------------------------------

            df_s = pd.read_csv(
                s_file,
                encoding="latin1",
            )

            df_v = pd.read_csv(
                v_file,
                encoding="latin1",
            )

            # Remove accidental whitespace
            df_s.columns = (
                df_s.columns
                .astype(str)
                .str.strip()
            )

            df_v.columns = (
                df_v.columns
                .astype(str)
                .str.strip()
            )

            # ---------------------------------------------------------
            # Locate IMU columns
            # ---------------------------------------------------------

            (
                accel_x,
                accel_y,
                accel_z,
                gyro_yaw,
                gyro_pitch,
                gyro_roll,
            ) = find_imu_columns(df_s)

            # ---------------------------------------------------------
            # Print first successful mapping
            # ---------------------------------------------------------

            if file_idx == 1:
                print("\nDetected IMU columns:")
                print(
                    f"  Accelerometer X : {accel_x}"
                )
                print(
                    f"  Accelerometer Y : {accel_y}"
                )
                print(
                    f"  Accelerometer Z : {accel_z}"
                )
                print(
                    f"  Gyroscope Yaw   : {gyro_yaw}"
                )
                print(
                    f"  Gyroscope Pitch : {gyro_pitch}"
                )
                print(
                    f"  Gyroscope Roll  : {gyro_roll}"
                )

            # ---------------------------------------------------------
            # Convert IMU values to numeric
            # ---------------------------------------------------------

            ax = pd.to_numeric(
                df_s[accel_x],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            ay = pd.to_numeric(
                df_s[accel_y],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            az = pd.to_numeric(
                df_s[accel_z],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            gy = pd.to_numeric(
                df_s[gyro_yaw],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            gp = pd.to_numeric(
                df_s[gyro_pitch],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            gr = pd.to_numeric(
                df_s[gyro_roll],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            # ---------------------------------------------------------
            # Raw 6-channel IMU
            #
            # IMPORTANT:
            # [ax, ay, az, gy, gp, gr]
            # ---------------------------------------------------------

            raw_imu = np.stack(
                [
                    ax,
                    ay,
                    az,
                    gy,
                    gp,
                    gr,
                ],
                axis=0,
            )

            # ---------------------------------------------------------
            # Ground-truth vehicle speed
            # ---------------------------------------------------------

            speed_column = find_speed_column(
                df_v
            )

            speed_kmh = pd.to_numeric(
                df_v[speed_column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            # ---------------------------------------------------------
            # Synchronize lengths
            # ---------------------------------------------------------

            min_len = min(
                raw_imu.shape[1],
                len(speed_kmh),
            )

            if min_len < (
                window_size + seq_len
            ):
                print(
                    f"\nSkipping short file:"
                    f"\n  {s_file}"
                    f"\n  samples={min_len}"
                )
                continue

            raw_imu = raw_imu[
                :,
                :min_len,
            ]

            speed_kmh = speed_kmh[
                :min_len
            ]

            # ---------------------------------------------------------
            # Align IMU to vehicle frame
            # ---------------------------------------------------------

            aligned_imu = (
                align_imu_to_vehicle_frame(
                    raw_imu
                )
            )

            # ---------------------------------------------------------
            # Convert speed to m/s
            # ---------------------------------------------------------

            speed_mps = (
                speed_kmh / 3.6
            ).astype(
                np.float32
            )

            # ---------------------------------------------------------
            # Remove invalid rows
            # ---------------------------------------------------------

            valid_rows = (
                np.isfinite(
                    aligned_imu
                ).all(axis=0)
                & np.isfinite(speed_mps)
            )

            aligned_imu = aligned_imu[
                :,
                valid_rows,
            ]

            speed_mps = speed_mps[
                valid_rows
            ]

            min_len = min(
                aligned_imu.shape[1],
                len(speed_mps),
            )

            if min_len < (
                window_size + seq_len
            ):
                continue

            # ---------------------------------------------------------
            # Build 32-sample spectral windows
            # ---------------------------------------------------------

            drive_windows = []
            drive_targets = []

            for start_idx in range(
                0,
                min_len - window_size + 1,
            ):

                end_idx = (
                    start_idx
                    + window_size
                )

                window = aligned_imu[
                    :,
                    start_idx:end_idx,
                ]

                target = (
                    speed_mps[
                        end_idx - 1
                    ]
                )

                if not np.isfinite(
                    window
                ).all():
                    continue

                if not np.isfinite(
                    target
                ):
                    continue

                # -----------------------------------------------------
                # EXACT project feature extractor
                #
                # 6 raw IMU channels
                #        ↓
                # 16 physics/spectral channels
                # -----------------------------------------------------

                features = (
                    compute_spectral_physics_features(
                        window
                    )
                )

                if features.shape != (
                    16,
                    window_size,
                ):
                    raise RuntimeError(
                        "Unexpected feature shape: "
                        f"{features.shape}. "
                        f"Expected "
                        f"(16, {window_size})."
                    )

                if not np.isfinite(
                    features
                ).all():
                    continue

                drive_windows.append(
                    features
                )

                drive_targets.append(
                    float(target)
                )

            if len(drive_windows) < seq_len:
                continue

            drive_windows = np.asarray(
                drive_windows,
                dtype=np.float32,
            )

            drive_targets = np.asarray(
                drive_targets,
                dtype=np.float32,
            )

            # ---------------------------------------------------------
            # Create temporal sequences
            #
            # Shape:
            #     (seq_len, 16, 32)
            # ---------------------------------------------------------

            file_sequence_count = 0

            for seq_start in range(
                0,
                len(drive_windows)
                - seq_len
                + 1,
                step_size,
            ):

                seq_end = (
                    seq_start
                    + seq_len
                )

                sequences.append(
                    (
                        drive_windows[
                            seq_start:seq_end
                        ],
                        drive_targets[
                            seq_start:seq_end
                        ],
                    )
                )

                file_sequence_count += 1

            if (
                file_idx % 10 == 0
                or file_idx == len(s_files)
            ):
                print(
                    f"  Processed "
                    f"{file_idx}/{len(s_files)} "
                    f"files | "
                    f"Sequences added: "
                    f"{file_sequence_count:,}"
                )

        except Exception as exc:

            print(
                "\nERROR loading:"
            )
            print(
                f"  {s_file}"
            )
            print(
                f"Exception type: "
                f"{type(exc).__name__}"
            )
            print(
                f"Exception: "
                f"{repr(exc)}"
            )

    print(
        f"\nTotal Driver E sequences: "
        f"{len(sequences):,}"
    )

    return sequences


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main(
    data_dir: str,
    checkpoint_path: str,
    batch_size: int = 256,
):

    device = torch.device(
        "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    # -------------------------------------------------------------
    # Build EXACT V2 NO-PRIOR architecture
    # -------------------------------------------------------------

    model = RecurrentSpeedFilterNet(
        in_channels=16,
        window_size=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        use_prior_speed=False,
    ).to(device)

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Support either:
    #   1. raw state_dict
    #   2. checkpoint dictionary
    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # -------------------------------------------------------------
    # Load Driver E
    # -------------------------------------------------------------

    sequences = (
        load_driver_e_sequences(
            data_dir=data_dir,
            window_size=32,
            seq_len=16,
            step_size=8,
        )
    )

    if not sequences:

        raise RuntimeError(
            "No valid Driver E sequences "
            "were found."
        )

    # -------------------------------------------------------------
    # Stack into arrays
    # -------------------------------------------------------------

    all_x = np.stack(
        [
            item[0]
            for item in sequences
        ],
        axis=0,
    )

    all_targets = np.stack(
        [
            item[1]
            for item in sequences
        ],
        axis=0,
    )

    print(
        f"\nInput tensor: "
        f"{all_x.shape}"
    )

    print(
        f"Target tensor: "
        f"{all_targets.shape}"
    )

    expected_shape = (
        16,
        16,
        32,
    )

    if all_x.shape[1:] != (
        16,
        16,
        32,
    ):

        raise RuntimeError(
            "Unexpected input shape: "
            f"{all_x.shape[1:]}. "
            f"Expected "
            f"{expected_shape}."
        )

    # -------------------------------------------------------------
    # Batched inference
    # -------------------------------------------------------------

    print(
        f"\nRunning batched inference "
        f"(batch_size={batch_size})..."
    )

    all_pred = []
    all_var = []

    total = len(all_x)

    with torch.no_grad():

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            x = torch.from_numpy(
                all_x[
                    start:end
                ]
            ).to(device)

            # NO prior-speed conditioning
            output, _ = model(
                x,
                v_prior=None,
            )

            output = (
                output
                .cpu()
                .numpy()
            )

            # output:
            # (B, Seq, 2)
            #
            # [:,:,0] = speed
            # [:,:,1] = variance

            all_pred.append(
                output[:, :, 0]
            )

            all_var.append(
                output[:, :, 1]
            )

            if (
                start == 0
                or end == total
                or (
                    start
                    // batch_size
                ) % 10 == 0
            ):

                print(
                    f"  Inference: "
                    f"{end:,}/{total:,}"
                )

    pred = np.concatenate(
        all_pred,
        axis=0,
    )

    var = np.concatenate(
        all_var,
        axis=0,
    )

    gt = all_targets

    # -------------------------------------------------------------
    # Convert to km/h
    # -------------------------------------------------------------

    gt_kmh = gt * 3.6
    pred_kmh = pred * 3.6

    # -------------------------------------------------------------
    # Uncertainty
    # -------------------------------------------------------------

    pred_std = np.sqrt(
        np.maximum(
            var,
            0.0,
        )
    )

    pred_std_kmh = (
        pred_std * 3.6
    )

    # -------------------------------------------------------------
    # Flatten for global metrics
    # -------------------------------------------------------------

    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)
    var_flat = var.reshape(-1)

    gt_kmh_flat = (
        gt_flat * 3.6
    )

    pred_kmh_flat = (
        pred_flat * 3.6
    )

    pred_std_flat = np.sqrt(
        np.maximum(
            var_flat,
            0.0,
        )
    )

    # -------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------

    errors = (
        pred_flat
        - gt_flat
    )

    errors_kmh = (
        pred_kmh_flat
        - gt_kmh_flat
    )

    # -------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------

    rmse_mps = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mae_mps = float(
        np.mean(
            np.abs(errors)
        )
    )

    bias_mps = float(
        np.mean(errors)
    )

    rmse_kmh = (
        rmse_mps * 3.6
    )

    mae_kmh = (
        mae_mps * 3.6
    )

    bias_kmh = (
        bias_mps * 3.6
    )

    # -------------------------------------------------------------
    # R²
    # -------------------------------------------------------------

    ss_res = np.sum(
        (
            gt_flat
            - pred_flat
        ) ** 2
    )

    ss_tot = np.sum(
        (
            gt_flat
            - np.mean(gt_flat)
        ) ** 2
    )

    if ss_tot > 1e-12:

        r2 = float(
            1.0
            - ss_res / ss_tot
        )

    else:

        r2 = float("nan")

    # -------------------------------------------------------------
    # Pearson correlation
    # -------------------------------------------------------------

    if (
        np.std(gt_flat) > 1e-12
        and np.std(pred_flat) > 1e-12
    ):

        correlation = float(
            np.corrcoef(
                gt_flat,
                pred_flat,
            )[0, 1]
        )

    else:

        correlation = float("nan")

    # -------------------------------------------------------------
    # Uncertainty coverage
    # -------------------------------------------------------------

    coverage = float(
        np.mean(
            np.abs(errors)
            <= 2.0
            * pred_std_flat
        )
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 82
    )

    print(
        "RECURRENT V2 — "
        "NO-PRIOR DRIVER E "
        "EVALUATION"
    )

    print(
        "=" * 82
    )

    print(
        f"Total evaluated sequence steps: "
        f"{gt_flat.size:,}"
    )

    print(
        f"Total evaluated sequences: "
        f"{len(sequences):,}"
    )

    print(
        f"RMSE: "
        f"{rmse_mps:.3f} m/s "
        f"({rmse_kmh:.2f} km/h)"
    )

    print(
        f"MAE:  "
        f"{mae_mps:.3f} m/s "
        f"({mae_kmh:.2f} km/h)"
    )

    print(
        f"Mean ground-truth speed: "
        f"{np.mean(gt_kmh_flat):.2f} km/h"
    )

    print(
        f"Mean predicted speed: "
        f"{np.mean(pred_kmh_flat):.2f} km/h"
    )

    print(
        f"Mean prediction bias: "
        f"{bias_mps:.3f} m/s "
        f"({bias_kmh:+.2f} km/h)"
    )

    print(
        f"R²: "
        f"{r2:.4f}"
    )

    print(
        f"Pearson correlation: "
        f"{correlation:.4f}"
    )

    print(
        f"Average predicted std dev: "
        f"{np.mean(pred_std_flat) * 3.6:.2f} km/h"
    )

    print(
        f"2-sigma coverage: "
        f"{coverage * 100.0:.2f}%"
    )

    print(
        "-" * 82
    )

    # -------------------------------------------------------------
    # Speed-bin evaluation
    # -------------------------------------------------------------

    print(
        f"{'Speed bin':<18} | "
        f"{'Count':>8} | "
        f"{'Mean GT':>10} | "
        f"{'Mean pred':>11} | "
        f"{'MAE':>9} | "
        f"{'Bias':>10}"
    )

    print(
        "-" * 82
    )

    for low, high in BINS:

        mask = (
            (gt_kmh_flat >= low)
            & (gt_kmh_flat < high)
        )

        if not np.any(mask):

            print(
                f"{low}-{high} km/h"
                f"{'':<8} | "
                f"{0:>8} | "
                f"{'n/a':>10} | "
                f"{'n/a':>11} | "
                f"{'n/a':>9} | "
                f"{'n/a':>10}"
            )

            continue

        bin_gt = (
            gt_kmh_flat[mask]
        )

        bin_pred = (
            pred_kmh_flat[mask]
        )

        bin_errors = (
            bin_pred
            - bin_gt
        )

        print(
            f"{low}-{high} km/h"
            f"{'':<8} | "
            f"{np.sum(mask):>8} | "
            f"{np.mean(bin_gt):>10.2f} | "
            f"{np.mean(bin_pred):>11.2f} | "
            f"{np.mean(np.abs(bin_errors)):>9.2f} | "
            f"{np.mean(bin_errors):>+10.2f}"
        )

    print(
        "=" * 82
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Recurrent V2 "
            "NO-prior model on held-out "
            "Driver E."
        )
    )

    parser.add_argument(
        "--data_dir",
        default="ml/data/IO-VNBD",
        help="Root directory of IO-VNBD dataset.",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "ml/weights/"
            "best_recurrent_v2_no_prior.pt"
        ),
        help="Model checkpoint.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Inference batch size.",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )