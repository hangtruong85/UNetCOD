"""
Model registry - centralized model definitions and creation
Provides easy import and instantiation of all available models
"""

import torch
import torch.nn as nn

# ===================== Model Imports =====================

# UNet++ models
from models.unetpp import (
    UNetPP,
    UNetPP_B3,
    UNetPP_Resnet50,
    UNetPP_B3_BEM,
)

# UNet models
from models.unet import (
    UNet,
    UNet_B3,
    UNet_Resnet50,
    UNet_B3_BEM,
)

# UNet3+ models
from models.unet3plus import (
    UNet3Plus,
    UNet3Plus_ResNet50,
    UNet3Plus_B0,
    UNet3Plus_B1,
    UNet3Plus_B2,
    UNet3Plus_B3,
    UNet3Plus_B4,
    UNet3Plus_B5,
    UNet3Plus_B3_BEM,
    UNet3Plus_B3_CBAM,
    UNet3Plus_B3_BEM_CBAM
)

# ===================== Model Dictionary =====================

MODEL_DICT = {     
    # UNet++ models
    "UNetPP": UNetPP,
    "UNetPP_B3": UNetPP_B3,
    "UNetPP_Resnet50": UNetPP_Resnet50,
    "UNetPP_B3_BEM": UNetPP_B3_BEM,
    
    # UNet models
    "UNet": UNet,
    "UNet_B3": UNet_B3,
    "UNet_Resnet50": UNet_Resnet50,
    "UNet_B3_BEM": UNet_B3_BEM,
    
    # UNet3+ baseline models
    "UNet3Plus": UNet3Plus, 
    "UNet3Plus_ResNet50": UNet3Plus_ResNet50,
    "UNet3Plus_B0": UNet3Plus_B0,
    "UNet3Plus_B1": UNet3Plus_B1,
    "UNet3Plus_B2": UNet3Plus_B2,
    "UNet3Plus_B3": UNet3Plus_B3,
    "UNet3Plus_B4": UNet3Plus_B4,
    "UNet3Plus_B5": UNet3Plus_B5,
    
    # UNet3+ with BEM
    "UNet3Plus_B3_BEM": UNet3Plus_B3_BEM,
    "UNet3Plus_B3_CBAM": UNet3Plus_B3_CBAM,
    "UNet3Plus_B3_BEM_CBAM": UNet3Plus_B3_BEM_CBAM
}


# ===================== Model Creation Function =====================

def create_model(model_name: str, device: str = "cuda") -> nn.Module:
    """
    Create and instantiate a model by name
    
    Args:
        model_name: Name of the model from MODEL_DICT
        device: Device to move model to ("cuda" or "cpu")
    
    Returns:
        model: Instantiated model on specified device
    
    Raises:
        ValueError: If model_name is not found in MODEL_DICT
    
    Example:
        >>> model = create_model("UNet3Plus_B3", device="cuda")
        >>> model = create_model("UNet3Plus_B3_BEM", device="cuda")
    """
    if model_name not in MODEL_DICT:
        available_models = "\n  - ".join(list(MODEL_DICT.keys()))
        raise ValueError(
            f"Unknown model: '{model_name}'\n"
            f"Available models:\n  - {available_models}"
        )
    
    model_class = MODEL_DICT[model_name]
    model = model_class().to(device)
    
    return model


def get_available_models() -> list:
    """
    Get list of all available model names
    
    Returns:
        list: Available model names
    """
    return list(MODEL_DICT.keys())


def print_available_models():
    """
    Print all available models in a formatted way
    """
    print("="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    
    # Categorize models
    models_by_category = {
        "UNet++": [k for k in MODEL_DICT.keys() if k.startswith("UNetPP")],
        "UNet": [k for k in MODEL_DICT.keys() if k.startswith("UNet") and not k.startswith("UNetPP") and not k.startswith("UNet3")],
        "UNet3+ (Baseline)": [k for k in MODEL_DICT.keys() if k.startswith("UNet3Plus") and "BEM" not in k and "PVT" not in k],
        "UNet3+ (with BEM)": [k for k in MODEL_DICT.keys() if k.startswith("UNet3Plus") and "BEM" in k],
    }
    
    for category, models in models_by_category.items():
        if models:
            print(f"\n{category}:")
            for model_name in sorted(models):
                print(f"  - {model_name}")
    
    print("\n" + "="*80)


# ===================== Model Info =====================

def get_model_info(model_name: str) -> dict:
    """
    Get information about a model
    
    Args:
        model_name: Name of the model
    
    Returns:
        dict: Model information (parameters count, architecture, etc.)
    """
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = create_model(model_name, device="cpu")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    has_bem = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    info = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "has_boundary_prediction": has_bem,
    }
    
    return info


def print_model_info(model_name: str):
    """
    Print detailed information about a model
    
    Args:
        model_name: Name of the model
    """
    try:
        info = get_model_info(model_name)
        
        print("="*80)
        print(f"Model: {info['model_name']}")
        print("="*80)
        print(f"Total Parameters      : {info['total_parameters']:,}")
        print(f"Trainable Parameters  : {info['trainable_parameters']:,}")
        print(f"Boundary Prediction   : {info['has_boundary_prediction']}")
        print("="*80)
    
    except ValueError as e:
        print(f"Error: {e}")