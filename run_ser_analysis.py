import os
import argparse
import torch
import torch.nn as nn

from export_result import save_results_to_json, check_if_run_exists

from model_utils.evaluate_model import evaluate_top1
from model_utils.quantize_model import (replace_layers_with_quant, calibrate_model)
from model_utils.model_info import create_fp_model, export_model
from model_utils.prepare_cifar10 import load_cifar10_data
from model_utils.prepare_cifar100 import load_cifar100_data
from soft_errors_utils.soft_error_injection import apply_ser_to_model


def main():
    parser = argparse.ArgumentParser(description = "Evaluate DNN model under soft errors.")

    parser.add_argument(
        "--model_name", type = str, default = 'resnet18', help = "Name of the model that we will evaluate."
    )
    parser.add_argument(
        "--dataset", type = str, default = 'cifar10', help = "Dataset used for Evaluation."
    )
    parser.add_argument(
        "--data_type", type = str, default = 'int8', choices = ['int8', 'int4', 'fp16','fp32'], 
        help = "Data type for quantization."
    )
    parser.add_argument(
        "--bit_idx", type = int, default = None, help = "Specific bit index for soft errors (None means that errors will be injected in all bits)."
    )
    parser.add_argument(
        "--ber", 
        type = float, 
        default = 1e-4,
        help = "Bit Error Rate (BER) for soft errors."
    )
    parser.add_argument(
        "--random_seed", type = int, default = 42, help = "Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        type = bool,
        default = False,
        help = 'Enable verbose logging during error injection.',
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dnn_model_name = args.model_name

    if args.bit_idx is None:
        if check_if_run_exists(args, f'results/{args.model_name}_{args.dataset}_ber_results.json'):
            return
        
    print(f"Running {dnn_model_name} on {device}")
    model = create_fp_model(dnn_model = dnn_model_name, dataset = args.dataset, device = device)

    if args.dataset == 'cifar10':
        _, test_loader, calib_loader = load_cifar10_data(
            batch_size = 128, val_ratio = 0.01
        )

    elif args.dataset == 'cifar100':
        _, test_loader, calib_loader = load_cifar100_data(
            batch_size = 128, val_ratio = 0.01
        )

    model.to(device)
    fp_accuracy = evaluate_top1(model, test_loader, device = device)
    print(f"Floating Point Model Accuracy: {fp_accuracy:.2f}%")

    if args.data_type == 'fp32':
        bit_width = 32
    elif args.data_type == 'fp16':
        bit_width = 16
    elif args.data_type == 'int8':
        bit_width = 8
    elif args.data_type == 'int4':
        bit_width = 4

    is_quant = args.data_type in ['int8', 'int4']

    if is_quant:
        print("Quantizing with data type", args.data_type)
    
    STORE_DIR = 'stored_models'
    MODEL_PATH = os.path.join(STORE_DIR, f"{dnn_model_name}_{args.dataset}_{args.data_type}.pt")

    if is_quant:
        if os.path.exists(MODEL_PATH):
            print(f"Loading quantized model from {MODEL_PATH}")
            quantized_model = replace_layers_with_quant(model, bit_width = bit_width, calibrated = True, 
                                                        verbose = False).to(device)
            quantized_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        else:
            quantized_model = replace_layers_with_quant(model, bit_width = bit_width, calibrated = False, 
                                                        verbose = False).to(device)
            calibrate_model(quantized_model, calibration_loader = calib_loader, device = device)
            export_model(quantized_model, MODEL_PATH)

        quantized_model.to(device)
        acc_clean = evaluate_top1(quantized_model, test_loader, device = device)
        print(f"Accuracy of Quantized Model without Errors: {acc_clean:.2f}%")
    else:
        # Floating-point path (fp32 / fp16)
        
        # Ensure model is in FP32 initially for speed
        model = model.float()
        
        if args.data_type == 'fp16':
            print(f"Simulating float16 precision (Pseudo-FP16)...")
            # Iterate over all parameters and truncate precision to FP16
            # Iterate over all modules (layers) in the model
            for module in model.modules():
                # Check if the module is a Convolution or Linear layer
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    for param in module.parameters():
                        if param.requires_grad:
                        # Precision truncation: fp32 -> fp16
                            param.data = param.data.half()
            # Note: BatchNorm running stats should usually stay high precision 
            # for stability, so we generally don't truncate .running_mean/var
            print(f"Model converted to float16 (weights)")          
        elif args.data_type == 'fp32':
            print(f"Model converted to float32 (single precision)")

        model.to(device)

        if args.verbose:
            print("Evaluating clean model before error injection...")

        acc_clean = evaluate_top1(model, test_loader, device = device)
        print(f"Accuracy of {args.data_type.upper()} Model without Errors: {acc_clean:.2f}%")

    model.to(torch.device('cpu'))

    corrupted_model = apply_ser_to_model(
        model = quantized_model if is_quant else model,
        soft_error_rate = args.ber,
        bit_width = bit_width,
        data_type = args.data_type,
        bit_idx = args.bit_idx,
        target_layers = None,
        random_seed = args.random_seed,
        verbose = args.verbose,
    )

    model.to(device)

    acc_corrupted = evaluate_top1(corrupted_model, test_loader, device = device)
    print(f"Accuracy with BER = {args.ber}: {acc_corrupted:.2f}%")
    save_results_to_json(args, fp_accuracy, acc_clean, acc_corrupted)
    return
    
if __name__ == "__main__":
    main()