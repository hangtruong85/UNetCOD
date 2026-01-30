import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from datasets.mhcd_dataset import MHCDDataset
from models.unetpp import (
    UNetPP,
    UNetPP_B3,
    UNetPP_Resnet50,
    UNetPP_B3_BEM,
)
from models.unet import (
    UNet,
    UNet_B3,
    UNet_Resnet50,
    UNet_B3_BEM,
)
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
    UNet3Plus_PVT_V2_B1,
    UNet3Plus_PVT_V2_B2
)
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.f_measure_paper import f_measure
from metrics.fweighted_measure import fw_measure
from logger_utils import setup_logger


# =========================================================
# Configuration
# =========================================================
class EvalConfig:
    """Evaluation configuration"""
    def __init__(self):
        # Model to evaluate
        self.model_name = "UNet3Plus_PVT_V2_B1"
        
        # Checkpoint path
        self.ckpt_path = "logs/UNet3Plus_PVT_V2_B1_20260130_111854/best_s_measure.pth"
        
        # Dataset
        self.root = "../MHCD_seg"
        self.split = "test"  # or "val"
        self.img_size = 352
        self.batch_size = 12
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output
        self.save_dir = f"eval_results/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)


# =========================================================
# Model Factory
# =========================================================
def create_model(model_name, device):
    """
    Create model based on name
    """
    models = {
        "UNetPP": UNetPP,
        "UNetPP_B3": UNetPP_B3,
        "UNetPP_Resnet50": UNetPP_Resnet50,
        "UNetPP_B3_BEM": UNetPP_B3_BEM,
        "UNet": UNet,
        "UNet_B3": UNet_B3,
        "UNet_Resnet50": UNet_Resnet50,
        "UNet_B3_BEM": UNet_B3_BEM,
        "UNet3Plus": UNet3Plus, 
        "UNet3Plus_ResNet50": UNet3Plus_ResNet50,
        "UNet3Plus_B0": UNet3Plus_B0,
        "UNet3Plus_B1": UNet3Plus_B1,
        "UNet3Plus_B2": UNet3Plus_B2,
        "UNet3Plus_B3": UNet3Plus_B3,
        "UNet3Plus_B4": UNet3Plus_B4,
        "UNet3Plus_B5": UNet3Plus_B5,
        "UNet3Plus_B3_BEM": UNet3Plus_B3_BEM,
        "UNet3Plus_PVT_V2_B1": UNet3Plus_PVT_V2_B1,
        "UNet3Plus_PVT_V2_B2": UNet3Plus_PVT_V2_B2
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name](n_classes=1).to(device)
    return model


# =========================================================
# Checkpoint Loading
# =========================================================
def load_checkpoint(model, ckpt_path, device):
    """
    Smart checkpoint loading - handles different formats
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Case 1: Checkpoint is a dictionary with 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "unknown")
        best_metric = ckpt.get("best_s_measure", "unknown")
        print(f"  ✓ Loaded from epoch {epoch}, best S-measure: {best_metric}")
    
    # Case 2: Checkpoint is raw state_dict
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        print(f"  ✓ Loaded state_dict")
    
    else:
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
    
    return model


# =========================================================
# Evaluation Metrics
# =========================================================
class MetricsAccumulator:
    """Accumulate metrics across batches"""
    def __init__(self):
        self.metrics = {
            "S": [],      # S-measure (Structure)
            "E": [],      # E-measure (Enhanced-alignment)
            #"F": [],      # F-measure (F-beta)
            "Fw": [],     # Weighted F-measure
            "MAE": []     # Mean Absolute Error
        }
    
    def update(self, pred, target):
        """
        Update metrics for a single sample
        Args:
            pred: predicted probability map (C, H, W)
            target: ground truth mask (C, H, W)
        """
        self.metrics["S"].append(s_measure(pred, target).item())
        self.metrics["E"].append(e_measure(pred, target).item())
        #self.metrics["F"].append(f_measure(pred, target).item())
        self.metrics["Fw"].append(fw_measure(pred, target).item())
        self.metrics["MAE"].append(torch.abs(pred - target).mean().item())
    
    def get_summary(self):
        """Get mean and std of all metrics"""
        summary = {}
        for key, values in self.metrics.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
        return summary
    
    def get_detailed(self):
        """Get all individual values"""
        return self.metrics


# =========================================================
# Evaluation Function
# =========================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate model on a dataset
    """
    model.eval()
    accumulator = MetricsAccumulator()
    
    logger.info("Starting evaluation...")
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Convert to probabilities
        pred_probs = torch.sigmoid(outputs)
        
        # Compute metrics for each sample in batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            accumulator.update(pred_probs[i], masks[i])
        
        # Update progress bar with current average
        current_metrics = accumulator.get_summary()
        pbar.set_postfix({
            'S': f"{current_metrics['S_mean']:.4f}",
            'MAE': f"{current_metrics['MAE_mean']:.4f}"
        })
    
    return accumulator


# =========================================================
# Results Display and Saving
# =========================================================
def display_results(summary, logger):
    """
    Display evaluation results in a nice format
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics_order = ['S', 'E', 'Fw', 'MAE']
    
    for metric in metrics_order:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        #print(f"{metric:12s}: {mean:.4f} ± {std:.4f}")
        print(f"{metric:8s}: {mean:.4f}")
        logger.info(f"{metric:12s}: {mean:.4f} ± {std:.4f}")
    
    print("="*60 + "\n")


def save_results(summary, detailed, config):
    """
    Save evaluation results to JSON files
    """
    # Save summary
    summary_path = os.path.join(config.save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(config.save_dir, "detailed.json")
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=4)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # Save config
    config_dict = {
        "model_name": config.model_name,
        "checkpoint": config.ckpt_path,
        "dataset_root": config.root,
        "split": config.split,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(config.save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Config saved to: {config_path}")


# =========================================================
# Main Evaluation
# =========================================================
def main():
    # Initialize config
    config = EvalConfig()
    
    # Setup logger
    logger = setup_logger(config.save_dir, "evaluation.log")
    
    logger.info("="*80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model Name   : {config.model_name}")
    logger.info(f"Checkpoint   : {config.ckpt_path}")
    logger.info(f"Dataset Root : {config.root}")
    logger.info(f"Split        : {config.split}")
    logger.info(f"Image Size   : {config.img_size}")
    logger.info(f"Batch Size   : {config.batch_size}")
    logger.info(f"Device       : {config.device}")
    logger.info(f"Save Dir     : {config.save_dir}")
    logger.info("="*80)
    
    # Create model
    logger.info(f"\nCreating model: {config.model_name}")
    model = create_model(config.model_name, config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Load checkpoint
    model = load_checkpoint(model, config.ckpt_path, config.device)
    
    # Create dataset
    logger.info(f"\nLoading dataset: {config.split}")
    dataset = MHCDDataset(
        config.root,
        config.split,
        config.img_size,
        logger=logger
    )
    logger.info(f"Total samples: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("STARTING EVALUATION")
    logger.info("="*80 + "\n")
    
    accumulator = evaluate_model(model, dataloader, config.device, logger)
    
    # Get results
    summary = accumulator.get_summary()
    detailed = accumulator.get_detailed()
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*80 + "\n")
    display_results(summary, logger)
    
    # Save results
    save_results(summary, detailed, config)
    
    logger.info(f"\nAll results saved to: {config.save_dir}")


# =========================================================
# Batch Evaluation - Compare Multiple Models
# =========================================================
def compare_models():
    """
    Evaluate and compare multiple DCN versions + new CBAM models
    """
    models_to_compare = [
        # Baseline
        ("UNetPP_B3", "logs/UNetPP_B3/best_s_measure.pth"),
        
        # DCN only
        ("UNetPP_DCNv1_COD", "logs/UNetPP_DCNv1_COD/best_s_measure.pth"),
        ("UNetPP_DCNv2_COD", "logs/UNetPP_DCNv2_COD/best_s_measure.pth"),
        ("UNetPP_DCNv3_COD", "logs/UNetPP_DCNv3_COD/best_s_measure.pth"),
        ("UNetPP_DCNv4_COD", "logs/UNetPP_DCNv4_COD/best_s_measure.pth"),
        
        # UNet3+ variants
        ("UNet3Plus_B3", "logs/UNet3Plus_B3/best_s_measure.pth"),
        ("UNet3Plus_DCNv2_COD", "logs/UNet3Plus_DCNv2_COD/best_s_measure.pth"),
        
        # NEW: Full models with CBAM
        ("UNetPP_DCNv1_CBAM_BEM", "logs/UNetPP_DCNv1_CBAM_BEM/best_s_measure.pth"),
        ("UNetPP_DCNv2_CBAM_BEM", "logs/UNetPP_DCNv2_CBAM_BEM/best_s_measure.pth"),
        ("UNetPP_DCNv3_CBAM_BEM", "logs/UNetPP_DCNv3_CBAM_BEM/best_s_measure.pth"),
        ("UNetPP_DCNv4_CBAM_BEM", "logs/UNetPP_DCNv4_CBAM_BEM/best_s_measure.pth"),
        
        # NEW: Ablation models
        ("UNetPP_CBAM", "logs/UNetPP_CBAM/best_s_measure.pth"),
        ("UNetPP_CBAM_BEM", "logs/UNetPP_CBAM_BEM/best_s_measure.pth"),
        ("UNetPP_DCN_BEM", "logs/UNetPP_DCN_BEM/best_s_measure.pth"),
    ]
    
    config = EvalConfig()
    logger = setup_logger("eval_results/comparison", "comparison.log")
    
    # Create dataset once
    dataset = MHCDDataset(config.root, config.split, config.img_size, logger=logger)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=4)
    
    results_comparison = {}
    
    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80 + "\n")
    
    for model_name, ckpt_path in models_to_compare:
        if not os.path.exists(ckpt_path):
            print(f"⚠ Skipping {model_name}: checkpoint not found")
            logger.warning(f"Skipping {model_name}: checkpoint not found at {ckpt_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")
        
        # Create and load model
        model = create_model(model_name, config.device)
        model = load_checkpoint(model, ckpt_path, config.device)
        
        # Evaluate
        accumulator = evaluate_model(model, dataloader, config.device, logger)
        summary = accumulator.get_summary()
        
        # Store results
        results_comparison[model_name] = summary
        
        # Display
        print(f"\nResults for {model_name}:")
        for key in ['S_mean', 'E_mean', 'Fw_mean', 'MAE_mean']:
            print(f"  {key:12s}: {summary[key]:.4f}")
    
    # Save comparison
    comparison_path = "eval_results/comparison/comparison_results.json"
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w') as f:
        json.dump(results_comparison, f, indent=4)
    
    print(f"\n✓ Comparison results saved to: {comparison_path}")
    logger.info(f"Comparison results saved to: {comparison_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<25} {'S-measure':<12} {'E-measure':<12} {'Fw-measure':<12} {'MAE':<12}")
    print("-"*80)
    
    for model_name, metrics in results_comparison.items():
        print(f"{model_name:<25} "
              f"{metrics['S_mean']:<12.4f} "
              f"{metrics['E_mean']:<12.4f} "
              f"{metrics['Fw_mean']:<12.4f} "
              f"{metrics['MAE_mean']:<12.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Compare multiple models
        compare_models()
    else:
        # Evaluate single model
        main()