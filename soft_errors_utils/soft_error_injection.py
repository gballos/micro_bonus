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

        bit_check_tensor = torch.rand(total_bits, device=device)
        flip_mask_bool = (bit_check_tensor < soft_error_rate)
        num_bitflips = int(flip_mask_bool.sum().item())

        flat = int_tensor.view(-1).clone()
        elems_changed = 0

        flips_applied = 0
        for _ in range(num_bitflips):
            idx = random.randint(0, flat.numel() - 1)
            if bit_idx is not None and bit_idx != "all":
                if bit_idx == 3 and bit_width == 4:
                    for b in range(3, 8):
                        if fault_model == 'bitflip':
                            flat[idx] ^= (1 << b) 
                        ## elif add more fault models if needed
                    flips_applied += 1
                else:
                    flat[idx] &= (1 << bit_idx)
                    flips_applied += 1
            else:
                b = random.randint(0, bit_width - 1)
                if b == 3 and bit_width == 4:
                    for bb in range(3, 8):
                        if fault_model == 'bitflip':
                            flat[idx] ^= (1 << bb) 
                         ## elif add more fault models if needed
                    flips_applied += 1
                else:
                    if fault_model == 'bitflip':
                        flat[idx] ^= ~(1 << b)
                     ## elif add more fault models if needed
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


def bitflip_int_tensor(
    int_tensor: torch.Tensor,
    bit_width: int,
    ber: float,
    bit_idx: Optional[int] = None,
    mantissa_only: bool = True,
) -> Tuple[torch.Tensor, int]:
    
    '''
    Apply bit-flip errors to an integer tensor based on the specified bit error rate (BER).

    :param int_tensor: Weight tensor in integer representation (int32, uint16, etc.)
    :param bit_width: Bit width of each element in the tensor (8, 16, 32)
    :param ber: Soft error rate (bit error rate)
    :param bit_idx: Specific bit index to flip, if None random bits are flipped
    :param mantissa_only: For float32/float16, flip only mantissa bits (more realistic)
                          This avoids catastrophic changes from exponent/sign flips
    :return: Tuple of (modified tensor, number of bits flipped)

    '''

    flat = int_tensor.view(-1)

    total_bits = flat.numel() * bit_width
    bit_mask = torch.rand(total_bits, device=flat.device) < ber
    bit_indices = torch.where(bit_mask)[0]

    if bit_indices.numel() == 0:
        return int_tensor, 0

    elem_idx = (bit_indices // bit_width).long()
    bit_idx_rand = (bit_indices % bit_width).long()

    # For mantissa-only flips in float32: bits 0-22 are mantissa, 23-30 are exponent, 31 is sign
    # For float16: bits 0-9 are mantissa, 10-14 are exponent, 15 is sign
    if mantissa_only and bit_width == 32:
        # Keep only mantissa bits (0-22)
        valid_mantissa = bit_idx_rand < 23
        elem_idx = elem_idx[valid_mantissa]
        bit_idx_rand = bit_idx_rand[valid_mantissa]
    elif mantissa_only and bit_width == 16:
        # Keep only mantissa bits (0-9)
        valid_mantissa = bit_idx_rand < 10
        elem_idx = elem_idx[valid_mantissa]
        bit_idx_rand = bit_idx_rand[valid_mantissa]

    # Bounds check to prevent out-of-range access
    valid_mask = elem_idx < flat.numel()
    elem_idx = elem_idx[valid_mask]
    bit_idx_rand = bit_idx_rand[valid_mask]

    flips_count = 0
    for e, b in zip(elem_idx, bit_idx_rand):
        e_val = e.item() if isinstance(e, torch.Tensor) else e
        b_val = b.item() if isinstance(b, torch.Tensor) else b
        
        if e_val >= flat.numel() or b_val >= bit_width:
            continue
        
        if bit_idx is None:
            flat[e_val] ^= (1 << b_val)
        else:
            if bit_idx < bit_width:
                flat[e_val] ^= (1 << bit_idx)
        flips_count += 1

    return int_tensor, flips_count

def error_injection_to_fp_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-6,   # BER
    target_layers=None,
    include_linear=False,
    seed=None,
    verbose=False,
    mantissa_only: bool = True,
    data_type: str = 'fp32',
):
    """
    Inject soft errors into floating-point model weights.
    
    :param model: The model to corrupt
    :param soft_error_rate: Bit error rate (fraction of bits to flip)
    :param target_layers: Specific layers to target
    :param include_linear: Include linear layers
    :param seed: Random seed for reproducibility
    :param verbose: Print detailed statistics
    :param mantissa_only: If True, only flip mantissa bits (realistic + avoids catastrophic exponent flips)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    eligible = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if not include_linear and isinstance(module, torch.nn.Linear):
                continue
            eligible.append((name, module))

    if target_layers is not None:
        if not isinstance(target_layers, Iterable):
            target_layers = [target_layers]
        if any(isinstance(t, str) for t in target_layers):
            name_set = set(t for t in target_layers if isinstance(t, str))
            eligible = [(n, m) for (n, m) in eligible if n in name_set]
        else:
            idx_set = set(int(t) for t in target_layers)
            eligible = [itm for i, itm in enumerate(eligible) if i in idx_set]

    report = []

    for lname, tgt in eligible:
        w = tgt.weight.data
        clean_float = w.clone()

        if data_type == 'fp32':
            int_view = w.view(torch.int32)
            bit_width = 32
        elif data_type == 'fp16':
            # For float16 (IEEE 754 half-precision), use uint16 to properly interpret bits
            # float16 layout: 1 sign bit, 5 exponent bits, 10 mantissa bits
            int_view = w.view(torch.uint16)
            bit_width = 16
        else:
            continue  # skip non-FP

        # Apply bit flips with mantissa-only option
        _, bits_flipped = bitflip_int_tensor(
            int_tensor=int_view,
            bit_width=bit_width,
            ber=soft_error_rate,
            mantissa_only=mantissa_only,
        )


        # Calculate actual weight change magnitude
        weight_diff = torch.abs(tgt.weight.data - clean_float)
        max_weight_change = weight_diff.max().item()
        mean_weight_change = weight_diff.mean().item()
        elems_changed = torch.count_nonzero(weight_diff > 0).item()
        
        report.append((lname, bit_width, elems_changed))

        if verbose:
            print(f"[FP] {lname}:")
            print(f"      dtype={data_type}, bit_width={bit_width}")
            print(f"      bits_flipped={bits_flipped}, elems_with_change={elems_changed}")
            print(f"      max_weight_change={max_weight_change:.6e}, mean_weight_change={mean_weight_change:.6e}")
            print(f"      weight_range=[{w.min().item():.6e}, {w.max().item():.6e}]")

    return report


def apply_ser_to_model(
        model, 
        soft_error_rate, 
        bit_width = 8, 
        data_type: str = 'int8',
        bit_idx = None, 
        include_linear = True,
        target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
        random_seed: Optional[int] = None,
        verbose: bool = False,
        mantissa_only: bool = True): 
    """
    Apply soft error injection to model weights.
    
    :param mantissa_only: For FP models, only flip mantissa bits (realistic, avoids catastrophic exponent flips)
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
    if data_type in ['int8', 'int4']:
        print("Starting soft error injection to quantized model weights...")
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
    if data_type in ['fp16', 'fp32']:
        print("Starting soft error injection to floating-point model weights...")
        error_injection_to_fp_model_weights(
            model = model,
            soft_error_rate = soft_error_rate,
            target_layers = target_layers,
            include_linear = include_linear,
            seed = random_seed,
            verbose = verbose,
            mantissa_only = mantissa_only,
            data_type = data_type,
        )
    print("Soft error injection completed.")
    return model

