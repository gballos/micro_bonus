#!/usr/bin/env python3
"""
Orchestrator for BER (Bit Error Rate) sweep.
Compatible with run_ser_analysis_stuck_up.py
"""

import argparse
import subprocess
import sys
from tqdm import tqdm
import numpy as np

# =======================
# Default Configuration
# =======================
DEFAULT_MODEL = "resnet18"
DEFAULT_DATASET = "cifar10"
DEFAULT_WORKER_SCRIPT = "run_ser_analysis_stuck_up.py"
RANDOM_SEED = 42

# =======================
# BER sweep values
# =======================
BER_RANGES = np.array([
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    1e-7, 2e-7, 5e-7,
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2
])

DATA_TYPES = ["int4", "int8", "fp16", "fp32"]
FP_BITS = ["mantissa", "exponent", "sign"]
STUCK_VALS = [0, 1]


# =======================
# Argument parsing
# =======================
def parse_args():
    parser = argparse.ArgumentParser( description="Run comprehensive BER sweep analysis")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--script", type=str, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


# =======================
# Run single job
# =======================
def run_job(script_path ,model_name ,dataset ,data_type ,ber ,seed ,stuck_val ,bit_idx="all",):
    cmd = [
        sys.executable,
        script_path,
        "--model_name", model_name,
        "--dataset", dataset,
        "--data_type", data_type,
        "--ber", str(ber),
        "--random_seed", str(seed),
        "--stuck_val", str(stuck_val),
        "--bit_idx", str(bit_idx),
    ]

    try:
        
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip().split("\n")[-1] if e.stderr else "Unknown error"
        return False, msg
    except Exception as e:
        return False, str(e)


# =======================
# Main sweep
# =======================
def main():
    args = parse_args()

    tasks = [(dt, ber) for dt in DATA_TYPES for ber in BER_RANGES]

    print(f"Starting BER sweep")
    print(f"Model   : {args.model_name}")
    print(f"Dataset : {args.dataset}")
    print(f"Jobs    : {len(tasks)}")
    print("-" * 60)

    with tqdm(
        total=len(tasks),
        unit="job",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:

        for data_type, ber in tasks:
            pbar.set_description(f"{data_type} @ {ber:.1e}")

            # Floating point → per bit
            if data_type in ["fp16", "fp32"]:
                for bit in FP_BITS:
                    for stuck_val in STUCK_VALS:
                        success, err = run_job(
                            args.script,
                            args.model_name,
                            args.dataset,
                            data_type,
                            ber,
                            args.seed,
                            stuck_val=stuck_val,
                            bit_idx=bit,
                        )
                        if not success:
                            tqdm.write(
                                f"❌ {data_type} {bit} stuck={stuck_val} ber={ber:.1e} | {err}"
                            )

            # Integer → all bits
            else:
                for stuck_val in STUCK_VALS:
                    success, err = run_job(
                        args.script,
                        args.model_name,
                        args.dataset,
                        data_type,
                        ber,
                        args.seed,
                        stuck_val=stuck_val,
                        bit_idx="all",
                    )
                    if not success:
                        tqdm.write(
                            f"❌ {data_type} stuck={stuck_val} ber={ber:.1e} | {err}"
                        )

            pbar.update(1)

    print("-" * 60)
    print("✅ Sweep Complete")


if __name__ == "__main__":
    main()
