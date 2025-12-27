import os
import json
import argparse

def save_results_to_json(args, fp_accuracy, acc_clean, acc_corrupted, prefix = ''):
   #if args.bit_idx is None:
    folder_name = 'results' if prefix == '' else prefix

    output_filename = f'{folder_name}/{args.model_name}_{args.dataset}_ber_results.json'
    
    run_result = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "ber": args.ber,
        "data_type": args.data_type,
        "bit_idx": args.bit_idx if args.bit_idx is not None else "all",
        "accuracy_fp": fp_accuracy,
        "accuracy_quantized": acc_clean,
        "accuracy_corrupted": acc_corrupted,
        "random_seed": args.random_seed
    }

    results_data = []
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r') as f:
                results_data = json.load(f)
                if not isinstance(results_data, list):
                    results_data = []
        except json.JSONDecodeError:
            results_data = []
    
    results_data.append(run_result)

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_filename, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\nSuccessfully saved results to {output_filename}")
    except IOError as e:
        print(f"\nError saving results to JSON: {e}")

def check_if_run_exists(args: argparse.Namespace, results_file: str = "soft_error_rates.json") -> bool:
    if not os.path.exists(results_file):
        return False

    # Define the required keys for matching
    REQUIRED_KEYS = {
        'model_name': 'model_name',
        'data_type': 'data_type',
        'ber': 'ber',
        'random_seed': 'random_seed',
        'bit_idx': 'bit_idx'
    }

    target_params = {}
    if args.bit_idx is None:
        args.bit_idx = "all"
    for arg_key, json_key in REQUIRED_KEYS.items():
        value = getattr(args, arg_key, None)
        target_params[json_key] = value 
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        for entry in data:
            match = True
            for json_key, target_value in target_params.items():
                entry_value = entry.get(json_key)
                if entry_value != target_value:
                    match = False
                    print(entry_value, target_value)
                    break
            
            if match:
                print(f"Skipping run: Result for Model={args.model_name}, DType={args.data_type}, BER={args.ber} already exists.")
                return True
                
    except json.JSONDecodeError:
        print("Warning: Could not read existing results file (JSONDecodeError). Proceeding with run.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while checking results file: {e}. Proceeding with run.")
        
    return False