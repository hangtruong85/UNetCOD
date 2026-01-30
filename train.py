import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
    UNet3Plus_PVT_V2_B2,
    UNet3Plus_PVT_V2_B3
)
from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from logger_utils import setup_logger


# ===================== Configuration =====================

class Config:
    """Centralized configuration for training"""
    def __init__(self):
        # Model selection - now with DCN + CBAM + BEM variants!
        self.model_name = "UNet3Plus_PVT_V2_B2"  # New full model
        
        # Dataset
        self.root = "../MHCD_seg"
        self.img_size = 352  # Increased from 256 - better for COD
        
        # Training
        self.epochs = 120
        self.batch_size = 12  # Reduced due to larger image size
        self.num_workers = 4
        
        # Optimizer
        self.lr_encoder = 1e-5  # Lower LR for pretrained encoder
        self.lr_dcn = 1e-4      # Higher LR for DCN modules
        self.weight_decay = 1e-4

        # Loss weights
        self.lambda_bce = 0.0
        self.lambda_dice = 0.4
        self.lambda_iou = 0.0
        self.lambda_focal = 0.4  # NEW: Focal loss weight
        self.lambda_boundary = 0.3
        
        if "BEM" in self.model_name:
            self.lambda_boundary = 0.3
            self.lambda_dice = 0.0
            self.lambda_focal = 0.0
            self.lambda_bce = 0.35
            self.lambda_iou = 0.35
        else:
            self.lambda_boundary = 0.0
            self.lambda_dice = 0.0
            self.lambda_focal = 0.0
            self.lambda_bce = 0.5
            self.lambda_iou = 0.5
        
        # Focal loss parameters
        self.focal_alpha = 0.25  # Weight for positive class
        self.focal_gamma = 2.0   # Focusing parameter
        
        # Training strategy
        self.warmup_epochs = 0  # Freeze encoder for first N epochs
        self.warmup_boundary_epochs = 30  # No boundary loss for first N epochs (stabilize segmentation first)
        self.use_cosine_schedule = True
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logging
        self.log_dir = f"logs/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)


# ===================== Boundary Extraction =====================

# ===================== Boundary Extraction (REMOVED) =====================
# BoundaryEnhancementModule đã được tích hợp trong model
# Models hiện tại chỉ return mask prediction, không có boundary prediction branch
# Nên không cần extract_boundary() function ở đây


# ===================== Loss Functions =====================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               gamma=0 -> equivalent to BCE
               gamma>0 -> down-weights easy examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth (B, 1, H, W), values in [0, 1]
        """
        # Get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate p_t
        # p_t = p if y=1, else 1-p
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Calculate alpha_t
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal weight
        focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class SegmentationLoss(nn.Module):
    """
    Combined loss for camouflaged object detection
    Includes: BCE + Dice + IoU + Focal + Boundary loss
    """
    def __init__(self, lambda_bce=1.0, lambda_dice=1.0, lambda_iou=0.5, 
                 lambda_focal=1.0, lambda_boundary=0.3,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_iou = lambda_iou
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def dice_loss(self, pred, target):
        """Dice loss for binary segmentation"""
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def iou_loss(self, pred, target):
        """IoU loss"""
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
    
    def boundary_loss(self, boundary_pred, boundary_target):
        """
        Boundary loss (L1 + BCE)
        Args:
            boundary_pred: predicted boundary logits (B, 1, H, W)
            boundary_target: ground truth boundary map (B, 1, H, W)
        """
        # L1 loss on probability space
        pred_prob = torch.sigmoid(boundary_pred)
        l1_loss = F.l1_loss(pred_prob, boundary_target)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(boundary_pred, boundary_target)
        
        return l1_loss + bce_loss
    
    def forward(self, pred, target, boundary_pred=None, boundary_target=None):
        """
        Args:
            pred: predicted mask logits (B, 1, H, W)
            target: ground truth mask (B, 1, H, W)
            boundary_pred: predicted boundary logits (B, 1, H, W) - optional
            boundary_target: ground truth boundary (B, 1, H, W) - optional
        """
        # Main segmentation losses
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice_loss(pred, target)
        loss_iou = self.iou_loss(pred, target)
        loss_focal = self.focal(pred, target)
        
        total_loss = (self.lambda_bce * loss_bce + 
                     self.lambda_dice * loss_dice + 
                     self.lambda_iou * loss_iou +
                     self.lambda_focal * loss_focal)
        
        # Add boundary loss if provided
        if boundary_pred is not None and boundary_target is not None:
            loss_boundary = self.boundary_loss(boundary_pred, boundary_target)
            total_loss += self.lambda_boundary * loss_boundary
        
        return total_loss


def mae_metric(pred, target):
    """Mean Absolute Error metric"""
    pred_prob = torch.sigmoid(pred)
    return torch.mean(torch.abs(pred_prob - target))


# ===================== Training Functions =====================

def train_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch):
    """
    Train for one epoch
    
    Training strategy:
    - Epoch 1-warmup_epochs: Freeze encoder
    - Epoch 1-warmup_boundary_epochs: No boundary loss (focus on segmentation)
    - Epoch warmup_boundary_epochs+: Full training with boundary loss
    """
    model.train()
    
    # Freeze encoder during warmup
    if epoch <= config.warmup_epochs:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
            for param in model.backbone.encoder.parameters():
                param.requires_grad = False
    else:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
            for param in model.backbone.encoder.parameters():
                param.requires_grad = True
    
    total_loss = 0.0
    num_batches = 0
    
    # Check if model supports boundary prediction
    has_boundary = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    # Determine if we should use boundary loss in this epoch
    use_boundary_loss = (epoch > config.warmup_boundary_epochs) and has_boundary
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            if use_boundary_loss:
                # Model can predict boundary
                mask_pred, boundary_pred = model(images, return_boundary=True)
                
                # Extract ground truth boundary from mask
                from models.boundary_enhancement import BoundaryEnhancementModule
                bem_temp = BoundaryEnhancementModule(channels=1).to(device)
                boundary_target = bem_temp.extract_boundary_map(masks)
                
                # Compute loss with boundary
                loss = criterion(mask_pred, masks, boundary_pred, boundary_target)
            else:
                # Model without boundary prediction
                mask_pred = model(images)
                loss = criterion(mask_pred, masks)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


# ===================== Validation Functions =====================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate model
    Returns dict with loss and metrics: S, E, F, MAE
    """
    model.eval()
    
    metrics = {
        "loss": 0.0,
        "S": 0.0,     # S-measure
        "E": 0.0,     # E-measure
        "Fw": 0.0,     # F-measure
        "MAE": 0.0    # Mean Absolute Error
    }
    
    num_samples = 0
    
    # Check if model supports boundary prediction
    has_boundary = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        if has_boundary:
            mask_pred, boundary_pred = model(images, return_boundary=True)
            
            # Extract ground truth boundary
            from models.boundary_enhancement import BoundaryEnhancementModule
            bem_temp = BoundaryEnhancementModule(channels=1).to(device)
            boundary_target = bem_temp.extract_boundary_map(masks)
            
            loss = criterion(mask_pred, masks, boundary_pred, boundary_target)
        else:
            mask_pred = model(images)
            loss = criterion(mask_pred, masks)
        
        # Convert to probabilities
        pred_probs = torch.sigmoid(mask_pred)
        
        # Compute metrics for each sample in batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            metrics["S"] += s_measure(pred_probs[i], masks[i]).item()
            metrics["E"] += e_measure(pred_probs[i], masks[i]).item()
            metrics["Fw"] += fw_measure(pred_probs[i], masks[i]).item()
            metrics["MAE"] += mae_metric(mask_pred[i:i+1], masks[i:i+1]).item()
        
        metrics["loss"] += loss.item() * batch_size
        num_samples += batch_size
    
    # Average metrics
    for key in metrics:
        metrics[key] /= num_samples
    
    return metrics


# ===================== Visualization =====================

def plot_training_curves(train_losses, val_losses, val_metrics, save_dir):
    """
    Plot training curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # S-measure
    s_measures = [m["S"] for m in val_metrics]
    axes[0, 1].plot(epochs, s_measures, label='S-measure', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('S-measure')
    axes[0, 1].set_title('S-measure (Structure Measure)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # E-measure and F-measure
    e_measures = [m["E"] for m in val_metrics]
    fw_measures = [m["Fw"] for m in val_metrics]
    axes[1, 0].plot(epochs, e_measures, label='E-measure', marker='o', color='blue')
    axes[1, 0].plot(epochs, fw_measures, label='Fw-measure', marker='s', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('E-measure and Fw-measure')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # MAE
    mae_scores = [m["MAE"] for m in val_metrics]
    axes[1, 1].plot(epochs, mae_scores, label='MAE', marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Mean Absolute Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


# ===================== Model Creation =====================

def create_model(model_name, device):
    """
    Create model based on name
    """
    model_dict = {     
        # UNet++ with CBAM and BEM
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
        "UNet3Plus_PVT_V2_B2": UNet3Plus_PVT_V2_B2,
        "UNet3Plus_PVT_V2_B3": UNet3Plus_PVT_V2_B3
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    model = model_dict[model_name]().to(device)
    return model


# ===================== Optimizer Creation =====================

def create_optimizer(model, config):
    """
    Create optimizer with different learning rates for different parts
    """
    # Separate encoder, DCN, CBAM, and other parameters
    encoder_params = []
    dcn_params = []
    cbam_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone.encoder' in name:
            encoder_params.append(param)
        elif 'dcn' in name.lower():
            dcn_params.append(param)
        elif 'cbam' in name.lower() or 'channel_attention' in name or 'spatial_attention' in name:
            cbam_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': config.lr_encoder,
            'name': 'encoder'
        })
        print(f"  Encoder params: {len(encoder_params)} tensors, LR={config.lr_encoder}")
    
    if dcn_params:
        param_groups.append({
            'params': dcn_params,
            'lr': config.lr_dcn,
            'name': 'dcn'
        })
        print(f"  DCN params: {len(dcn_params)} tensors, LR={config.lr_dcn}")
    
    if cbam_params:
        param_groups.append({
            'params': cbam_params,
            'lr': config.lr_dcn,  # Same as DCN
            'name': 'cbam'
        })
        print(f"  CBAM params: {len(cbam_params)} tensors, LR={config.lr_dcn}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.lr_dcn,
            'name': 'other'
        })
        print(f"  Other params: {len(other_params)} tensors, LR={config.lr_dcn}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def create_scheduler(optimizer, config, num_batches):
    """
    Create learning rate scheduler
    """
    if config.use_cosine_schedule:
        total_steps = config.epochs * num_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    
    return scheduler


# ===================== Main Training Loop =====================

def main():
    # Initialize configuration
    config = Config()
    
    # Setup logger
    logger = setup_logger(config.log_dir, "train.log")
    
    # Log configuration
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model Name    : {config.model_name}")
    logger.info(f"Dataset Root  : {config.root}")
    logger.info(f"Image Size    : {config.img_size}")
    logger.info(f"Batch Size    : {config.batch_size}")
    logger.info(f"Epochs        : {config.epochs}")
    logger.info(f"Warmup Epochs : {config.warmup_epochs}")
    logger.info(f"Warmup Boundary Epochs: {config.warmup_boundary_epochs}")
    logger.info(f"LR Encoder    : {config.lr_encoder}")
    logger.info(f"LR DCN        : {config.lr_dcn}")
    logger.info(f"Loss Weights  : BCE={config.lambda_bce}, Dice={config.lambda_dice}, "
                f"IoU={config.lambda_iou}, Focal={config.lambda_focal}, Boundary={config.lambda_boundary}")
    logger.info(f"Focal Params  : alpha={config.focal_alpha}, gamma={config.focal_gamma}")
    logger.info(f"Device        : {config.device}")
    logger.info(f"Log Directory : {config.log_dir}")
    logger.info("="*80)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MHCDDataset(config.root, "train", config.img_size, logger=logger)
    val_dataset = MHCDDataset(config.root, "val", config.img_size, logger=logger)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples  : {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating model: {config.model_name}")
    model = create_model(config.model_name, config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters    : {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer, criterion, scheduler
    optimizer = create_optimizer(model, config)
    criterion = SegmentationLoss(
        lambda_bce=config.lambda_bce,
        lambda_dice=config.lambda_dice,
        lambda_iou=config.lambda_iou,
        lambda_focal=config.lambda_focal,
        lambda_boundary=config.lambda_boundary,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma
    )
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler()
    
    # Training state
    start_epoch = 1
    best_s_measure = 0.0
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(config.log_dir, "last.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_s_measure = checkpoint["best_s_measure"]
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        val_metrics_history = checkpoint.get("val_metrics", [])
        
        logger.info(f"Resumed from epoch {start_epoch}, best S-measure: {best_s_measure:.4f}")
    
    # CSV logging
    csv_path = os.path.join(config.log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "S_measure", "E_measure", "fw_measure", "MAE"])
    
    # Training loop
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch}/{config.epochs}")
        logger.info(f"{'='*80}")
        
        # Training phase
        if epoch <= config.warmup_epochs:
            logger.info(f"[WARMUP] Encoder frozen, training DCN modules only")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, config.device, config, epoch)
        train_losses.append(train_loss)
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, config.device)
        val_losses.append(val_metrics["loss"])
        val_metrics_history.append(val_metrics)
        
        # Logging
        logger.info(f"[TRAIN] Loss: {train_loss:.4f}")
        logger.info(f"[VAL]   Loss: {val_metrics['loss']:.4f} | "
                   f"S: {val_metrics['S']:.4f} | "
                   f"E: {val_metrics['E']:.4f} | "
                   f"F: {val_metrics['Fw']:.4f} | "
                   f"MAE: {val_metrics['MAE']:.4f}")
        
        # Save to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["S"],
                val_metrics["E"],
                val_metrics["Fw"],
                val_metrics["MAE"]
            ])
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_s_measure": best_s_measure,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics_history,
        }
        torch.save(checkpoint, os.path.join(config.log_dir, "last.pth"))
        
        # Save best model
        if val_metrics["S"] > best_s_measure:
            best_s_measure = val_metrics["S"]
            torch.save(model.state_dict(), os.path.join(config.log_dir, "best_s_measure.pth"))
            logger.info(f"🔥 NEW BEST S-measure: {best_s_measure:.4f}")
        
        # Update learning rate
        if config.use_cosine_schedule:
            for _ in range(len(train_loader)):
                scheduler.step()
        else:
            scheduler.step()
        
        # Plot training curves
        if epoch % 5 == 0 or epoch == config.epochs:
            plot_training_curves(train_losses, val_losses, val_metrics_history, config.log_dir)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Best S-measure: {best_s_measure:.4f}")
    logger.info(f"Model saved to: {config.log_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()