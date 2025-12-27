import torch.nn as nn
import torch
from torch import optim

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode

def quantize_conv(fp_layer, bit_width = 8, dtype = 'int', calibrated = False, verbose = False):
    if verbose:
        print(f"[INFO]: Using {dtype} quantization for Conv2d layer.")

    return QuantConv2d(
        in_channels = fp_layer.in_channels,
        out_channels = fp_layer.out_channels,
        kernel_size = fp_layer.kernel_size,
        stride = fp_layer.stride,
        padding = fp_layer.padding,
        dilation = fp_layer.dilation,
        groups = fp_layer.groups,
        bias = (fp_layer.bias is not None) or calibrated,
        weight_bit_width = bit_width,
        weight_scaling_per_output_channel = False,
        weight_quant = Int8WeightPerTensorFixedPoint
    )

def quantize_linear(fp_layer, bit_width = 8, dtype = 'int', calibrated = False, verbose = False):
    if verbose:
        print(f"[INFO]: Using {dtype} quantization for Linear layer.")
    return QuantLinear(
        in_features = fp_layer.in_features,
        out_features = fp_layer.out_features,
        bias = (fp_layer.bias is not None) or calibrated,
        weight_bit_width = bit_width,
        weight_quant = Int8WeightPerTensorFixedPoint
    )

def quantize_activation(act_width = 8):
    return QuantReLU(bit_width = act_width, act_quant = Int8ActPerTensorFloat,)

def quantize_identity(act_width = 8):
    return QuantIdentity(act_quant = Int8ActPerTensorFloat, return_quant_tensor = True, 
                         bit_width = act_width)
    

def replace_layers_with_quant(module, bit_width: int = 8, calibrated: bool = False,
                              verbose = False) -> nn.Module:
    if isinstance(module, nn.Conv2d):
        quant_layer = quantize_conv(module, bit_width = bit_width, calibrated = calibrated,
                                    verbose = verbose)
        quant_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_layer.bias.data = module.bias.data.clone()
        return quant_layer

    elif isinstance(module, nn.Linear):
        quant_layer = quantize_linear(module, bit_width = bit_width, calibrated = calibrated,
                                      verbose = verbose)
        quant_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_layer.bias.data = module.bias.data.clone()
        return quant_layer

    elif isinstance(module, nn.ReLU):
        return quantize_activation(act_width = bit_width)
    
    elif isinstance(module, nn.Identity):
        return quantize_identity(act_width = bit_width)

    for name, child in module.named_children():
        setattr(module, name, replace_layers_with_quant(
            child, bit_width = bit_width,
            calibrated = calibrated,verbose = verbose
        ))
    return module

def calibrate_model(quant_model, calibration_loader, device = 'cpu'):
    with torch.no_grad():
        with calibration_mode(quant_model):
            for i, (images, _) in enumerate(calibration_loader):
                images = images.to(device)
                quant_model(images)

        with bias_correction_mode(quant_model):
            for i, (images, _) in enumerate(calibration_loader):
                images = images.to(device)
                quant_model(images)

    return quant_model


def train_quant_model(quant_net, train_loader, val_loader = None, device = 'cpu',
                      epochs = 20, lr = 0.0001):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quant_net.parameters(), lr = lr)

    patience = 5
    best_val_loss = float('inf')

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = quant_net(images.float())
            loss = criterion(log_ps, labels)

            #backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            val_loss = 0
            accuracy = 0

        # Turn off gradients for validation
        with torch.no_grad():
            quant_net.eval()
            if(val_loader != None):
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = quant_net(images.float())
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Check for early stopping
        if(val_loader != None):
            avg_val_loss = val_loss/len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

        quant_net.train()

    return quant_net