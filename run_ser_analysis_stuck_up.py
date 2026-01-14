import os
import argparse
import torch

from export_result import save_results_to_json, check_if_run_exists
from model_utils.evaluate_model import evaluate_top1
from model_utils.quantize_model import replace_layers_with_quant, calibrate_model
from model_utils.model_info import create_fp_model, export_model
from model_utils.prepare_cifar10 import load_cifar10_data
from model_utils.prepare_cifar100 import load_cifar100_data
from stuck_at_errors_utils.stuck_at_error_injection import apply_ser_to_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DNN model under STUCK-AT hardware faults"
    )

    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument(
    "--data_type",
    type=str,
    default="int8",
    choices=["int8", "int4", "fp16", "fp32"],
    help="Numeric representation used for model weights",
    )


    parser.add_argument(
    "--bit_idx",
    type=str,
    default="mantissa",
    help=(
        "INT: bit index (0–7) or 'all'. "
        "FP: 'sign', 'exponent', 'mantissa', or 'all'."
        ),
    )


    parser.add_argument(
        "--stuck_val",
        type=int,
        choices=[0, 1],
        required=True,
        help="0 = stuck-at-0, 1 = stuck-at-1",
    )

    parser.add_argument(
        "--ber",
        type=float,
        default=1e-4,
        help="Probability a bit becomes permanently stuck",
    )

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dnn_model_name = args.model_name
    print(f"Running {args.model_name} on {device}")
    model = create_fp_model(dnn_model = dnn_model_name, dataset = args.dataset, device = device)

    if args.dataset == "cifar10":
        _, test_loader, calib_loader = load_cifar10_data(
            batch_size=128, val_ratio=0.01
        )
    else:
        _, test_loader, calib_loader = load_cifar100_data(
            batch_size=128, val_ratio=0.01
        )

    model.to(device)

   
    fp_accuracy = evaluate_top1(model, test_loader, device=device)
    print(f"Floating-point accuracy: {fp_accuracy:.2f}%")

    if args.data_type == "fp32":
        bit_width = 32
    elif args.data_type == "fp16":
        bit_width = 16
    elif args.data_type == "int8":
        bit_width = 8
    elif args.data_type == "int4":
        bit_width = 4

    is_quant = args.data_type in ["int8", "int4"]


    if is_quant:
        print(f"Quantizing with data type {args.data_type}")
        
    STORE_DIR = "stored_models"
    MODEL_PATH = os.path.join(STORE_DIR,f"{dnn_model_name}_{args.dataset}_{args.data_type}.pt",)
    if is_quant:
        if os.path.exists(MODEL_PATH):
            print(f"Loading quantized model from {MODEL_PATH}")
            correct_model = replace_layers_with_quant( model,bit_width=bit_width,calibrated=True,verbose=False, ).to(device)
            correct_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        else:
            print("Calibrating quantized model...")
            correct_model = replace_layers_with_quant(model,bit_width=bit_width,calibrated=False,verbose=False,).to(device)
            calibrate_model(correct_model, calib_loader, device=device)
            export_model(correct_model, MODEL_PATH)
            
        correct_model.to(device)
        acc_clean = evaluate_top1( correct_model, test_loader, device=device)
        print(f"Accuracy of Quantized Model without Errors: {acc_clean:.2f}%")


    else:
        correct_model = model.float()

        if args.data_type == "fp16":
            if args.verbose:
                print("Converting model to FP16...")
            correct_model = correct_model.half()

        print(
            f"Using {args.data_type.upper()} model "
            f"(bit_width={bit_width})"
        )

        correct_model.to(device)

        acc_clean = evaluate_top1(
            correct_model, test_loader, device=device
        )
        print(
            f"Accuracy of {args.data_type.upper()} Model without Errors: "
            f"{acc_clean:.2f}%"
        )


    correct_model.to("cpu")

    
    corrupted_model = apply_ser_to_model(
    model=correct_model,
    soft_error_rate=args.ber,
    bit_width=bit_width,
    data_type=args.data_type,
    bit_idx=args.bit_idx,
    stuck_val=args.stuck_val,
    random_seed=args.random_seed,
    verbose=args.verbose,
    )


    
    corrupted_model.to(device)
    acc_corrupted = evaluate_top1(corrupted_model, test_loader, device=device)

    print(
        f"Accuracy after STUCK-AT faults "
        f"(stuck_val={args.stuck_val}, BER={args.ber}): "
        f"{acc_corrupted:.2f}%"
    )

    
    save_results_to_json(args, fp_accuracy, acc_clean, acc_corrupted)


if __name__ == "__main__":
    main()
