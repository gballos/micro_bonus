import torch
import random
from typing import Iterable, Optional, Union, List, Tuple

# Try importing Brevitas, handle graciously if missing (for non-quantized runs)
try:
    from brevitas.nn import QuantLinear
except ImportError:
    QuantLinear = None

def _reshape_scale_for_weight(scale: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Helper to align quantization scales with weight dimensions."""
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        if W.ndim == 4:
            return scale.view(-1, 1, 1, 1)
        elif W.ndim == 2:
            return scale.view(-1, 1)
    return scale 

def bitflip_int_tensor(
    int_tensor: torch.Tensor,
    bit_width: int,
    ber: float,
    bit_idx: Optional[Union[int, str]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Apply bit-flip errors to an integer tensor.
    
    :param int_tensor: The integer representation of the weights.
    :param bit_width: Total bits per element (e.g., 8, 16, 32).
    :param ber: Bit Error Rate.
    :param bit_idx: Specific bit to flip (0 to bit_width-1) or semantic string ('mantissa', 'all').
                    Accepts numeric strings like "3".
    """
    
    # 1. Normalize bit_idx
    # If it's a digit string like "3", convert to int 3.
    is_specific_bit = False
    if isinstance(bit_idx, str) and bit_idx.isdigit():
        bit_idx = int(bit_idx)
    
    if isinstance(bit_idx, int):
        is_specific_bit = True

    flat = int_tensor.view(-1)

    # 2. Generate Random Errors
    total_bits = flat.numel() * bit_width
    bit_mask = torch.rand(total_bits, device=flat.device) < ber
    bit_indices = torch.where(bit_mask)[0]

    if bit_indices.numel() == 0:
        return int_tensor, 0

    elem_idx = (bit_indices // bit_width).long()
    bit_idx_rand = (bit_indices % bit_width).long()

    # 3. Filter based on semantic bit_idx (e.g., "mantissa")
    if bit_idx == 'mantissa' and bit_width == 32:
        # FP32: Bits 0-22 are mantissa
        valid_mantissa = bit_idx_rand < 23
        elem_idx = elem_idx[valid_mantissa]
        bit_idx_rand = bit_idx_rand[valid_mantissa]
    elif bit_idx == 'mantissa' and bit_width == 16:
        # FP16: Bits 0-9 are mantissa
        valid_mantissa = bit_idx_rand < 10
        elem_idx = elem_idx[valid_mantissa]
        bit_idx_rand = bit_idx_rand[valid_mantissa]
    
    # Bounds check safety
    valid_mask = elem_idx < flat.numel()
    elem_idx = elem_idx[valid_mask]
    bit_idx_rand = bit_idx_rand[valid_mask]

    flips_count = 0
    
    # 4. Apply Flips
    # Using a loop for safety with mixed types, though vectorization is possible.
    # We iterate only over the indices selected for flipping.
    for e, b in zip(elem_idx, bit_idx_rand):
        e_val = e.item() if isinstance(e, torch.Tensor) else e
        b_val = b.item() if isinstance(b, torch.Tensor) else b
        
        if e_val >= flat.numel():
            continue

        if is_specific_bit:
            # Force flip at the user-specified index (ignoring the random bit index 'b')
            # Only flip if the specified bit is within the width (e.g. bit 30 in 8-bit is invalid)
            if bit_idx < bit_width:
                flat[e_val] ^= (1 << bit_idx)
                flips_count += 1
        else:
            # Flip the randomly selected bit (b_val)
            if b_val < bit_width:
                flat[e_val] ^= (1 << b_val)
                flips_count += 1

    return int_tensor, flips_count

def error_injection_to_quant_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-3,
    data_type: str = 'int8',
    bit_width: int = 8,
    bit_idx: Optional[Union[int, str]] = None,
    target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
    include_linear: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    fault_model: str = 'bitflip'
) -> List[Tuple[str, int, int]]:
    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Normalize bit_idx locally for logic that depends on it before bitflip function
    if isinstance(bit_idx, str) and bit_idx.isdigit():
        bit_idx = int(bit_idx)

    eligible: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        # Check for Brevitas QuantLinear or similar quantized layers
        if hasattr(module, 'quant_weight') and module.quant_weight is not None:
            if not include_linear and (QuantLinear and isinstance(module, QuantLinear)):
                continue
            eligible.append((name, module))

    # Filter target layers
    if target_layers is not None:
        if not isinstance(target_layers, Iterable) or isinstance(target_layers, str):
            target_layers = [target_layers]
        
        if any(isinstance(t, str) for t in target_layers):
            name_set = set(t for t in target_layers if isinstance(t, str))
            eligible = [(n, m) for (n, m) in eligible if n in name_set]
        else:
            idx_set = set(int(t) for t in target_layers)
            eligible = [itm for i, itm in enumerate(eligible) if i in idx_set]

    report: List[Tuple[str, int, int]] = []

    for _, (lname, tgt) in enumerate(eligible):
        device = tgt.weight.device
        
        # Get scale and quantized integer weights
        scale = tgt.weight_quant.scale().detach().to(device)
        scale_b = _reshape_scale_for_weight(scale, tgt.weight)
        
        # Brevitas usually returns float representation of quant weights; cast to int
        # Note: Ensure the model is actually in a quantized state where .int() is valid
        int_tensor = tgt.quant_weight().int().detach().to(device)
        
        if data_type == 'int8':
            int_tensor = int_tensor.to(torch.int8)
        elif data_type == 'int4':
            # PyTorch doesn't have native int4, usually stored as int8 or packed
            int_tensor = int_tensor.to(torch.int8)

        # Generate Error Mask
        num_elems = int_tensor.numel()
        total_bits = num_elems * bit_width
        bit_check_tensor = torch.rand(total_bits, device=device)
        flip_mask_bool = (bit_check_tensor < soft_error_rate)
        num_bitflips = int(flip_mask_bool.sum().item())

        flat = int_tensor.view(-1).clone()
        flips_applied = 0
        
        # Apply Flips (Manual loop for specific fault models in Quantized path)
        for _ in range(num_bitflips):
            idx = random.randint(0, flat.numel() - 1)
            
            # --- Specific Bit Index Logic ---
            if isinstance(bit_idx, int):
                # Custom logic: if targeting bit 3 in int4, flip MSBs (bits 3-7 in int8 container)
                if bit_idx == 3 and bit_width == 4:
                    for b in range(3, 8):
                        if fault_model == 'bitflip':
                            flat[idx] ^= (1 << b) 
                    flips_applied += 1
                else:
                    if bit_idx < bit_width:
                        flat[idx] ^= (1 << bit_idx)
                        flips_applied += 1
            
            # --- Random Bit Logic ---
            else:
                b = random.randint(0, bit_width - 1)
                
                # Custom logic: if random bit hits bit 3 in int4
                if b == 3 and bit_width == 4:
                    for bb in range(3, 8):
                        if fault_model == 'bitflip':
                            flat[idx] ^= (1 << bb) 
                    flips_applied += 1
                else:
                    if fault_model == 'bitflip':
                        flat[idx] ^= (1 << b)
                    flips_applied += 1

        # Restore weights
        if data_type in ['int8', 'int4']:
            flipped_tensor = flat.view_as(int_tensor)
        
        # Convert back to float representation (De-quantize)
        float_tensor = flipped_tensor.to(torch.float32) * scale_b

        # Update model weights in-place
        clean_float_tensor = tgt.quant_weight().detach().to(device)
        with torch.no_grad():
            tgt.weight.copy_(float_tensor.to(tgt.weight.dtype))

        # Logging
        elems_changed = 0
        if verbose:
            diff_mask = (clean_float_tensor != float_tensor)
            diff_indices = torch.where(diff_mask.flatten())[0]
            elems_changed = int(diff_mask.sum().item())

            print(f"=============== [INFO]: layer {lname} =================") 
            print(f"flips_applied={flips_applied}, elems_changed={elems_changed}")
            print(f"N={num_elems}, ser={soft_error_rate:.2e}, bit_width={bit_width}, bit_idx={bit_idx}")
            print(f"weights_shape={tgt.weight.shape}")

        report.append((lname, flips_applied, elems_changed))
        
    return report

def error_injection_to_fp_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-6,
    target_layers=None,
    include_linear=False,
    seed=None,
    verbose=False,
    bit_idx: Optional[Union[int, str]] = 'all',
    data_type: str = 'fp32',
):
    """
    Inject soft errors into floating-point model weights.
    Supports FP32, Native FP16, and Pseudo-FP16 (FP32 storage simulating FP16).
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Normalize bit_idx locally
    if isinstance(bit_idx, str) and bit_idx.isdigit():
        bit_idx = int(bit_idx)

    eligible = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if not include_linear and isinstance(module, torch.nn.Linear):
                continue
            eligible.append((name, module))

    if target_layers is not None:
        if not isinstance(target_layers, Iterable) or isinstance(target_layers, str):
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
        bits_flipped = 0
        bit_width = 32

        # --- FP32 Injection ---
        if data_type == 'fp32':
             if w.dtype != torch.float32: 
                 continue
             int_view = w.view(torch.int32)
             bit_width = 32
             _, bits_flipped = bitflip_int_tensor(int_view, bit_width, soft_error_rate, bit_idx)

        # --- Native FP16 Injection (GPU) ---
        elif data_type == 'fp16' and w.dtype == torch.float16:
             int_view = w.view(torch.int16)
             bit_width = 16
             _, bits_flipped = bitflip_int_tensor(int_view, bit_width, soft_error_rate, bit_idx)
             
        # --- Pseudo-FP16 Injection (CPU Safe) ---
        # Weights are stored as FP32, but we treat them as FP16 for error injection
        elif data_type == 'fp16' and w.dtype == torch.float32:
             if verbose:
                 print(f"pseudo-FP16 injection on {lname}")
                 
             # 1. Cast to Half to get correct bit structure
             w_half = w.half()
             
             # 2. View as Int16
             int_view = w_half.view(torch.int16)
             bit_width = 16
             
             # 3. Flip Bits
             _, bits_flipped = bitflip_int_tensor(int_view, bit_width, soft_error_rate, bit_idx)
             
             # 4. Cast back to FP32 and assign to model
             tgt.weight.data = w_half.float()
        
        else:
             # Skip incompatible layers (e.g. attempting FP16 on Int8 weights)
             continue

        # Statistics
        weight_diff = torch.abs(tgt.weight.data - clean_float)
        elems_changed = torch.count_nonzero(weight_diff > 0).item()
        
        report.append((lname, bit_width, elems_changed))
        
        if verbose:
            max_change = weight_diff.max().item() if elems_changed > 0 else 0.0
            mean_change = weight_diff.mean().item()
            
            print(f"=============== [INFO]: layer {lname} =================")
            print(f"dtype={data_type}, storage={w.dtype}, bit_width={bit_width}")
            print(f"bits_flipped={bits_flipped}, elems_changed={elems_changed}")
            print(f"max_weight_change={max_change:.6e}, mean_weight_change={mean_change:.6e}")

    return report

def apply_ser_to_model(
        model: torch.nn.Module, 
        soft_error_rate: float, 
        bit_width: int = 8, 
        data_type: str = 'int8',
        bit_idx: Optional[Union[int, str]] = None, 
        include_linear: bool = True,
        target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.nn.Module:
    """
    Main entry point for Soft Error Rate (SER) injection.
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    if data_type in ['int8', 'int4']:
        print(f"Starting soft error injection (Type: {data_type}, BER: {soft_error_rate})...")
        error_injection_to_quant_model_weights(
            model=model,
            soft_error_rate=soft_error_rate,
            bit_width=bit_width,
            data_type=data_type,
            bit_idx=bit_idx,
            include_linear=include_linear,
            target_layers=target_layers,
            seed=random_seed,
            verbose=verbose,
        )
        
    elif data_type in ['fp16', 'fp32']:
        print(f"Starting soft error injection (Type: {data_type}, BER: {soft_error_rate})...")
        error_injection_to_fp_model_weights(
            model=model,
            soft_error_rate=soft_error_rate,
            target_layers=target_layers,
            include_linear=include_linear,
            seed=random_seed,
            verbose=verbose,
            bit_idx=bit_idx,
            data_type=data_type,
        )
        
    print("Soft error injection completed.")
    return model