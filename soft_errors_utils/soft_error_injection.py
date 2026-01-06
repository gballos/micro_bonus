import torch
import random
from typing import Iterable, Optional, Union, List, Tuple

# Try importing Brevitas, handle graciously if missing
try:
    from brevitas.nn import QuantLinear
except ImportError:
    QuantLinear = None

def _reshape_scale_for_weight(scale: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 0: return scale
    if scale.ndim == 1:
        if W.ndim == 4: return scale.view(-1, 1, 1, 1)
        elif W.ndim == 2: return scale.view(-1, 1)
    return scale 

def bitflip_int_tensor(
    int_tensor: torch.Tensor,
    bit_width: int,
    ber: float,
    bit_idx: Optional[Union[int, str]] = None,
    use_ecc: bool = True  # SECDED Toggle
) -> Tuple[torch.Tensor, int]:
    """
    Applies bit-flips with SECDED simulation. 
    1 bit error per weight = Corrected (No flip)
    2+ bit errors per weight = Uncorrectable (Flipped)
    """
    is_specific_bit = False
    if isinstance(bit_idx, str) and bit_idx.isdigit():
        bit_idx = int(bit_idx)
    if isinstance(bit_idx, int):
        is_specific_bit = True

    flat = int_tensor.view(-1)
    total_bits = flat.numel() * bit_width
    
    # Generate potential flips
    bit_mask = torch.rand(total_bits, device=flat.device) < ber
    bit_indices = torch.where(bit_mask)[0]

    if bit_indices.numel() == 0:
        return int_tensor, 0

    # Group by element to check ECC capability
    elem_indices = (bit_indices // bit_width).long()
    unique_elems, flip_counts = torch.unique(elem_indices, return_counts=True)
    
    flips_applied = 0
    
    for i, e_idx in enumerate(unique_elems):
        num_errors = flip_counts[i].item()
        
        # --- ECC Logic ---
        # Single bit errors are fixed by SECDED
        if use_ecc and num_errors == 1:
            continue 
            
        # Get specific bits for this corrupted element
        local_bits = bit_indices[elem_indices == e_idx] % bit_width
        
        for b in local_bits:
            b_val = b.item()
            if is_specific_bit:
                if bit_idx < bit_width:
                    flat[e_idx] ^= (1 << bit_idx)
                    flips_applied += 1
            else:
                # Semantic filtering
                if bit_idx == 'mantissa':
                    if bit_width == 32 and b_val >= 23: continue # Skip Exp/Sign
                    if bit_width == 16 and b_val >= 10: continue
                
                flat[e_idx] ^= (1 << b_val)
                flips_applied += 1

    return int_tensor, flips_applied

def error_injection_to_quant_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-3,
    data_type: str = 'int8',
    bit_width: int = 8,
    bit_idx: Optional[Union[int, str]] = None,
    use_ecc: bool = True,
    target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
    include_linear: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> List[Tuple[str, int, int]]:
    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    eligible = []
    for name, module in model.named_modules():
        if hasattr(module, 'quant_weight') and module.quant_weight is not None:
            if not include_linear and (QuantLinear and isinstance(module, QuantLinear)):
                continue
            eligible.append((name, module))

    # Filter target layers
    if target_layers is not None:
        if isinstance(target_layers, (str, int)): target_layers = [target_layers]
        if any(isinstance(t, str) for t in target_layers):
            eligible = [(n, m) for (n, m) in eligible if n in target_layers]
        else:
            eligible = [itm for i, itm in enumerate(eligible) if i in target_layers]

    report = []
    for lname, tgt in eligible:
        device = tgt.weight.device
        scale = tgt.weight_quant.scale().detach().to(device)
        scale_b = _reshape_scale_for_weight(scale, tgt.weight)
        
        # Clone and cast to int container
        int_tensor = tgt.quant_weight().int().detach().to(device)
        
        # Apply Injection with ECC
        flipped_tensor, flips_applied = bitflip_int_tensor(
            int_tensor, bit_width, soft_error_rate, bit_idx, use_ecc=use_ecc
        )
        
        # De-quantize and Update
        float_tensor = flipped_tensor.to(torch.float32) * scale_b
        with torch.no_grad():
            tgt.weight.copy_(float_tensor.to(tgt.weight.dtype))

        report.append((lname, flips_applied, flips_applied)) # simplified report
        
        if verbose:
            print(f"Layer: {lname} | Flips (Post-ECC): {flips_applied}")
            
    return report

def error_injection_to_fp_model_weights(
    model: torch.nn.Module,
    soft_error_rate: float = 1e-6,
    data_type: str = 'fp32',
    bit_idx: Optional[Union[int, str]] = 'all',
    use_ecc: bool = True,
    target_layers=None,
    include_linear=False,
    seed=None,
    verbose=False,
):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    eligible = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if not include_linear and isinstance(module, torch.nn.Linear): continue
            eligible.append((name, module))

    # Filter target layers (Simplified)
    if target_layers is not None:
        eligible = [itm for i, itm in enumerate(eligible) if i in target_layers]

    report = []
    for lname, tgt in eligible:
        w = tgt.weight.data
        
        if data_type == 'fp32':
            int_view = w.view(torch.int32)
            _, flips = bitflip_int_tensor(int_view, 32, soft_error_rate, bit_idx, use_ecc)
        elif data_type == 'fp16':
            # Handle Pseudo-FP16 or Native
            if w.dtype == torch.float32:
                w_half = w.half()
                int_view = w_half.view(torch.int16)
                _, flips = bitflip_int_tensor(int_view, 16, soft_error_rate, bit_idx, use_ecc)
                tgt.weight.data = w_half.float()
            else:
                int_view = w.view(torch.int16)
                _, flips = bitflip_int_tensor(int_view, 16, soft_error_rate, bit_idx, use_ecc)
        
        report.append((lname, 32 if data_type=='fp32' else 16, flips))
        
    return report

def apply_ser_to_model(
        model: torch.nn.Module, 
        soft_error_rate: float, 
        bit_width: int = 8, 
        data_type: str = 'int8',
        bit_idx: Optional[Union[int, str]] = None, 
        use_ecc: bool = True,
        include_linear: bool = True,
        target_layers: Optional[Union[Iterable[str], Iterable[int]]] = None, 
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.nn.Module:
    
    print(f"Injecting errors... (Type: {data_type}, BER: {soft_error_rate}, ECC: {use_ecc})")
    
    if data_type in ['int8', 'int4']:
        error_injection_to_quant_model_weights(
            model, soft_error_rate, data_type, bit_width, bit_idx, use_ecc,
            target_layers, include_linear, random_seed, verbose
        )
    else:
        error_injection_to_fp_model_weights(
            model, soft_error_rate, data_type, bit_idx, use_ecc,
            target_layers, include_linear, random_seed, verbose
        )
        
    return model