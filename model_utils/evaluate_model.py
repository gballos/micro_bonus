import torch

def evaluate_top1(model, test_loader, device='cpu', evaluate_mode=True):
    if evaluate_mode:
        model.eval()
    
    correct = 0
    total = 0

    # Cache the model dtype to avoid repeated parameter access
    model_dtype = next(model.parameters()).dtype
    is_fp16 = (model_dtype == torch.float16)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Convert input to match model dtype
            if is_fp16:
                inputs = inputs.half()
            else:
                inputs = inputs.float()

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy
