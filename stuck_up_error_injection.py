import torch
import random
from typing import Iterable, Optional, Union, List, Tuple


def _reshape_scale_for_weight(scale: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        if W.ndim == 4:
            return scale.view(-1, 1, 1, 1)
        elif W.ndim == 2:
            return scale.view(-1, 1)
    return scale



def stuck_at_int_tensor(
    int_tensor: torch.Tensor,
    bit_width: int,
    ber: float,
    bit_idx: Union[int, str] = "all",
    stuck_val: int = 1,
) -> int:
    """
    Apply stuck-at faults to an integer tensor.

    stuck_val = 1 → stuck-at-1
    stuck_val = 0 → stuck-at-0
    """

    flat = int_tensor.view(-1)
    total_bits = flat.numel() * bit_width

    # Randomly choose faulty bits
    faulty_mask = torch.rand(total_bits, device=flat.device) < ber
    faulty_indices = torch.where(faulty_mask)[0]

    if faulty_indices.numel() == 0:
        return 0

    elem_idx = faulty_indices // bit_width
    bit_idx_rand = faulty_indices % bit_width

    # Select specific bit if requested
    if isinstance(bit_idx, int):
        keep = bit_idx_rand == bit_idx
        elem_idx = elem_idx[keep]
        bit_idx_rand = bit_idx_rand[keep]

    flips = 0
    for e, b in zip(elem_idx.tolist(), bit_idx_rand.tolist()):
        if stuck_val == 1:
            flat[e] |= (1 << b)       # force bit to 1
        else:
            flat[e] &= ~(1 << b)      # force bit to 0
        flips += 1

    return flips



def error_injection_to_quant_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float,
    bit_width: int,
    data_type: str,
    bit_idx: Union[int, str],
    stuck_val: int,
    target_layers=None,
    seed=None,
    verbose=False,
) -> List[Tuple[str, int]]:

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    report = []

    for lname, module in model.named_modules():
        if not hasattr(module, "quant_weight"):
            continue

        device = module.weight.device

        scale = module.weight_quant.scale().detach().to(device)
        scale_b = _reshape_scale_for_weight(scale, module.weight)

        # Integer weights
        int_tensor = module.quant_weight().int().detach().to(device)

        # Inject stuck-at faults
        flips = stuck_at_int_tensor(
            int_tensor=int_tensor,
            bit_width=bit_width,
            ber=soft_error_rate,
            bit_idx=bit_idx,
            stuck_val=stuck_val,
        )

        # Dequantize back
        float_tensor = int_tensor.float() * scale_b

        with torch.no_grad():
            module.weight.copy_(float_tensor)

        if verbose:
            print(f"[STUCK-AT][INT] Layer={lname}, stuck_val={stuck_val}, faults={flips}")

        report.append((lname, flips))

    return report



def apply_ser_to_model(
    model: torch.nn.Module,
    soft_error_rate: float,
    bit_width: int,
    data_type: str,
    bit_idx: Union[int, str],
    stuck_val: int,
    target_layers=None,
    random_seed=None,
    verbose=False,
):

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    
    if data_type in ["int8", "int4"]:
        error_injection_to_quant_model_weights(
            model=model,
            soft_error_rate=soft_error_rate,
            bit_width=bit_width,
            data_type=data_type,
            bit_idx=bit_idx,
            stuck_val=stuck_val,
            target_layers=target_layers,
            seed=random_seed,
            verbose=verbose,
        )

    
    elif data_type in ["fp16", "fp32"]:

        if data_type == "fp32":
            view_dtype = torch.int32
            bit_width = 32
        else:
            view_dtype = torch.int16
            bit_width = 16

        for lname, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue

           
            w = module.weight.data
            int_view = w.view(view_dtype)

            flips = stuck_at_int_tensor(
                int_tensor=int_view,
                bit_width=bit_width,
                ber=soft_error_rate,
                bit_idx=bit_idx,
                stuck_val=stuck_val,
            )

            if verbose and flips > 0:
                print(f"[STUCK-AT][FP] Layer={lname}, stuck_val={stuck_val}, faults={flips}")

    print(" Stuck-at fault injection completed.")
    return model
