#!/usr/bin/env python3
"""
Orchestrator for BER (Bit Error Rate) sweep.
Runs the evaluation script for various data types and error rates.
"""

import argparse
import subprocess
import sys
from tqdm import tqdm

# Default Configuration
DEFAULT_MODEL = "resnet18"
DEFAULT_DATASET = "cifar10"
DEFAULT_WORKER_SCRIPT = "run_ser_analysis.py" 
RANDOM_SEED = 42

# BER ranges to sweep
BER_RANGES = {
    'int8': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'int4': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'fp32': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    'fp16': [1e-7, 5e-7, 1e-6, 5e-6, 1e-5],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive BER sweep analysis.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Name of the model.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name.")
    parser.add_argument("--script", type=str, default=DEFAULT_WORKER_SCRIPT, help="Path to the worker script.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return parser.parse_args()

def run_job(script_path, model_name, dataset, data_type, ber, seed):
    """
    Executes the worker script for a specific configuration.
    Relies on the worker script's internal logic to skip if results exist.
    """
    cmd = [
        sys.executable,
        script_path,
        "--model_name", model_name,
        "--dataset", dataset,
        "--data_type", data_type,
        "--ber", str(ber),
        "--random_seed", str(seed),
        "--verbose", "False",
    ]

    try:
        # Run subprocess; suppress stdout/stderr to keep progress bar clean
        # If you need to debug, remove capture_output=True
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True  # Raises CalledProcessError on non-zero exit code
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip().split('\n')[-1] if e.stderr else "Unknown Error"
        return False, error_msg
    except Exception as e:
        return False, str(e)

def main():
    args = parse_args()

    # 1. Flatten the configuration list for the progress bar
    tasks = []
    for dtype, ber_list in BER_RANGES.items():
        for ber in ber_list:
            tasks.append((dtype, ber))

    print(f"Starting sweep for {args.model_name} on {args.dataset}")
    print(f"Total configurations to run: {len(tasks)}")
    print("-" * 60)

    # 2. Initialize Progress Bar
    # ncols=100 ensures the bar doesn't wrap weirdly in small terminals
    with tqdm(total=len(tasks), unit="job", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        for data_type, ber in tasks:
            # Update description to show what's currently running
            pbar.set_description(f"Running {data_type} @ {ber:.1e}")
            
            success, error_msg = run_job(
                args.script, 
                args.model_name, 
                args.dataset, 
                data_type, 
                ber, 
                args.seed
            )
            
            if not success:
                # If a job fails, we print above the progress bar so it persists
                tqdm.write(f"❌ Failed: {data_type} @ {ber:.1e} | Error: {error_msg}")
            
            # Since your main.py skips existing runs internally, 
            # this loop will just move very fast for finished jobs.
            pbar.update(1)

    print("-" * 60)
    print("Sweep Complete.")

if __name__ == "__main__":
    main()