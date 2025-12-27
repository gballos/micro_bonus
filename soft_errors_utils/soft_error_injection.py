import torch
import random
from brevitas.nn import QuantLinear
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


def error_injection_to_quant_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-3,
    data_type: str = 'int8',
    bit_width: int = 8,
    bit_idx: Optional[int] = None,
    target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
    include_linear: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    fault_model: str = 'bitflip'
) -> List[Tuple[str, int, int]]:

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    eligible: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, 'quant_weight') and module.quant_weight is not None:
            if not include_linear and isinstance(module, QuantLinear):
                continue
            eligible.append((name, module))

    if target_layers is not None:
        if not isinstance(target_layers, Iterable):
            target_layers = [target_layers]
        if any(isinstance(t, str) for t in target_layers):
            name_set = set(t for t in target_layers if isinstance(t, str))
            eligible = [(n, m) for (n, m) in eligible if n in name_set]
        else:
            # Indices
            idx_set = set(int(t) for t in target_layers)
            eligible = [itm for i, itm in enumerate(eligible) if i in idx_set]

    report: List[Tuple[str, int, int]] = []

    for _, (lname, tgt) in enumerate(eligible):
        device = tgt.weight.device

        scale = tgt.weight_quant.scale().detach().to(device)
        scale_b = _reshape_scale_for_weight(scale, tgt.weight)

        if data_type in ['int8', 'int4']:
            int_tensor = tgt.quant_weight().int().detach().to(device).to(torch.int8)

        num_elems = int_tensor.numel()
        total_bits = num_elems * bit_width

        # Per-bit Bernoulli sampling for flips: this ensures BER controls
        # the expected number of flipped bits precisely.
        bit_check_tensor = torch.rand(total_bits, device=device)
        flip_mask_bool = (bit_check_tensor < float(soft_error_rate))
        positions = torch.nonzero(flip_mask_bool, as_tuple=False).flatten()
        num_bitflips = int(positions.numel())

        flat = int_tensor.view(-1).clone()
        elems_changed = 0

        flips_applied = 0
        if num_bitflips > 0:
            # If a specific bit index is requested, filter positions accordingly
            if bit_idx is not None and bit_idx != "all":
                positions = positions[(positions % bit_width) == bit_idx]

            for pos in positions:
                pos_i = int(pos.item())
                elem_idx = pos_i // bit_width
                b = pos_i % bit_width

                # For standard bitflip model, flip the selected bit using XOR
                if fault_model == 'bitflip':
                    flat[elem_idx] = int(flat[elem_idx].item()) ^ (1 << int(b))
                flips_applied += 1

        if data_type in ['int8', 'int4']:
            flipped_tensor = flat.view_as(int_tensor)
                       
        float_tensor = flipped_tensor.to(torch.float32) * scale_b

        clean_float_tensor = tgt.quant_weight().detach().to(device)
        with torch.no_grad():
            tgt.weight.copy_(float_tensor.to(tgt.weight.dtype))

        if verbose:
            diff_mask = (clean_float_tensor != float_tensor).to(torch.float32)
            diff_indices = torch.where(diff_mask.flatten())[0]
            elems_changed = int(diff_mask.sum().item())

            print(f"=============== [INFO]: layer {lname} =================") 
            print(f"flips_applied={flips_applied}, elems_changed={elems_changed}, "
                  f"N={num_elems}, ser={soft_error_rate:g}, bit_width={bit_width}, bit_idx={bit_idx}, weight_dtype={tgt.weight.dtype}, "
                  f"weights_shape={tgt.weight.shape}")
            
            scale_b_flat = scale_b.flatten() 
            sample_indices = diff_indices[:10]
            print("   [Index] | Clean Value | Corrupted Value | Scale")
            print("   " + "-" * 48)
            clean_flat = clean_float_tensor.flatten()
            corrupted_flat = float_tensor.flatten()

            for i in sample_indices:
                current_scale = scale_b_flat.item() 
                idx = i.item()
                clean_val_full = clean_flat[0][idx].item()
                corrupted_val_full = corrupted_flat[idx].item()

                clean_val_scaled = clean_val_full / current_scale
                corrupted_val_scaled = corrupted_val_full / current_scale
                
                print(f"   [{i.item():<7}] | {clean_val_scaled:.8f} | {corrupted_val_scaled:.8f} | {current_scale}")

            report.append((lname, flips_applied, elems_changed))
    return report


def apply_ser_to_quant_model(
        model, 
        soft_error_rate, 
        bit_width = 8, 
        data_type: str = 'int8',
        bit_idx = None, 
        include_linear = True,
        target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
        random_seed: Optional[int] = None,
        verbose: bool = False): 
    
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    error_injection_to_quant_model_weights(
        model = model,
        soft_error_rate = soft_error_rate,
        bit_width = bit_width,
        data_type = data_type,
        bit_idx = bit_idx,
        include_linear = include_linear,
        target_layers = target_layers,
        seed = random_seed,
        verbose = verbose,
    )
    print("Soft error injection completed.")
    return model
