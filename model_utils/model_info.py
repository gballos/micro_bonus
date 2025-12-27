import os
import torch
import detectors ## it is needed to download the pretrained model from Hugging Face
import timm

def create_fp_model(dnn_model = 'resnet18', dataset = 'cifar10', device = 'cpu'):

    if dnn_model == 'resnet18' and dataset == 'cifar10':
        model = timm.create_model("resnet18_cifar10", pretrained = True)
    elif dnn_model == 'resnet18' and dataset == 'cifar100':
        model = timm.create_model("resnet18_cifar100", pretrained = True)
    elif dnn_model == 'resnet50' and dataset == 'cifar100':
        model = timm.create_model("resnet50_cifar100", pretrained = True)
    else:
        raise ValueError(f"Model '{dnn_model}' is not supported.")
        
    model = model.to(device)
    return model

def export_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving quantized model state to: {model_path}")
    torch.save(model.state_dict(), model_path)