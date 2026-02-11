import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import timm
from .boundary_enhancement import BoundaryEnhancementModule
from .cbam import CBAM

# ============================================================================
# UNet3+ Decoder Block
# ============================================================================
"""
UNet3+ Decoder Block for full-scale skip connections
Aggregates features from ALL encoder levels

According to original paper: https://arxiv.org/abs/2004.08790
Full-scale skip connection pattern:
- d4 = e1 + e2 + e3 + e4 + e5
- d3 = e1 + e2 + e3 + d4 + e5  
- d2 = e1 + e2 + d3 + d4 + e5  
- d1 = e1 + d2 + d3 + d4 + e5  

Key insight: e5 (deepest encoder) is used in ALL decoder levels
"""
class UNet3PlusDecoderBlock(nn.Module):
    """
    UNet3+ Decoder Block with full-scale skip connections
    Aggregates features from ALL encoder levels
    
    Args:
        in_channels_list: list of input channels from each encoder level
        out_channels: output channels after fusion
    
    Example:
        >>> decoder4 = UNet3PlusDecoderBlock(
        ...     in_channels_list=[64, 128, 320, 512],
        ...     out_channels=64
        ... )
    """
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: list of input channels from each encoder level
            out_channels: output channels after fusion
        """
        super().__init__()
        
        # Calculate channels after concatenation
        cat_channels = out_channels * len(in_channels_list)
        
        # Convs to reduce each encoder feature to out_channels
        self.conv_branches = nn.ModuleList()
        for in_ch in in_channels_list:
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_size):
        """
        Args:
            features: list of feature maps from all encoder levels
            target_size: (H, W) target size for this decoder level
        
        Returns:
            out: fused feature map of shape (B, out_channels, H, W)
        """
        processed = []
        
        for feat, conv in zip(features, self.conv_branches):
            # Resize to target size if needed
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Process with conv
            feat = conv(feat)
            processed.append(feat)
        
        # Concatenate and fuse
        cat_feat = torch.cat(processed, dim=1)
        out = self.fusion(cat_feat)
        
        return out
     
class UNet3Plus(nn.Module):
    """
    UNet3+ with baseline encoder 
    Full-scale skip connections for better feature aggregation
    
    CORRECT pattern according to original paper:
    - d4 = e1 + e2 + e3 + e4 + e5
    - d3 = e1 + e2 + e3 + d4 + e5
    - d2 = e1 + e2 + d3 + d4 + e5
    - d1 = e1 + d2 + d3 + d4 + e5
    """
    def __init__(self, n_classes=1, encoder="resnet34"):
        super().__init__()
        
        # Encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels
        encoder_channels = self.encoder.out_channels
        
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                encoder_channels[4],  # e4
                encoder_channels[5]   # e5
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                decoder_channels,     # d4 (NOT e4!)
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                decoder_channels,     # d3 (NOT e3!)
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                decoder_channels,     # d2 (NOT e2!)
                decoder_channels,     # d3
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        e1, e2, e3, e4, e5 = features[1], features[2], features[3], features[4], features[5]
        
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])  # Same as e4
        size_d3 = (e3.shape[2], e3.shape[3])  # Same as e3
        size_d2 = (e2.shape[2], e2.shape[3])  # Same as e2
        size_d1 = (e1.shape[2], e1.shape[3])  # Same as e1
        
        # Decoder 4: d4 = e1 + e2 + e3 + e4 + e5
        d4 = self.decoder4([e1, e2, e3, e4, e5], size_d4)
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        d3 = self.decoder3([e1, e2, e3, d4, e5], size_d3)
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        d2 = self.decoder2([e1, e2, d3, d4, e5], size_d2)
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        d1 = self.decoder1([e1, d2, d3, d4, e5], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask

class UNet3Plus_B3(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder
    Full-scale skip connections for better feature aggregation
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                encoder_channels[4],  # e4
                encoder_channels[5]   # e5
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                decoder_channels,     # d4 (NOT e4!)
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                decoder_channels,     # d3 (NOT e3!)
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                decoder_channels,     # d2 (NOT e2!)
                decoder_channels,     # d3
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        e1, e2, e3, e4, e5 = features[1], features[2], features[3], features[4], features[5]
        
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])  # Same as e4
        size_d3 = (e3.shape[2], e3.shape[3])  # Same as e3
        size_d2 = (e2.shape[2], e2.shape[3])  # Same as e2
        size_d1 = (e1.shape[2], e1.shape[3])  # Same as e1
        
        # Decoder 4: d4 = e1 + e2 + e3 + e4 + e5
        d4 = self.decoder4([e1, e2, e3, e4, e5], size_d4)
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        d3 = self.decoder3([e1, e2, e3, d4, e5], size_d3)
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        d2 = self.decoder2([e1, e2, d3, d4, e5], size_d2)
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        d1 = self.decoder1([e1, d2, d3, d4, e5], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask
        
class UNet3Plus_B3_x(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder (Baseline)
    Full-scale skip connections for better feature aggregation
    """
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b3")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_B0(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b0")
    def forward(self, x):
        return self.backbone(x)
        
class UNet3Plus_B1(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b1")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_B2(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b2")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_B4(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b4")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_B5(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="efficientnet-b5")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_ResNet50(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = UNet3Plus(encoder="resnet50")
    def forward(self, x):
        return self.backbone(x)

class UNet3Plus_B3_BEM(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder + BEM
    Full-scale skip connections for better feature aggregation
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__()
        
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                encoder_channels[4],  # e4
                encoder_channels[5]   # e5
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                encoder_channels[3],  # e3
                decoder_channels,     # d4 (NOT e4!)
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                encoder_channels[2],  # e2
                decoder_channels,     # d3 (NOT e3!)
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1
                decoder_channels,     # d2 (NOT e2!)
                decoder_channels,     # d3
                decoder_channels,     # d4
                encoder_channels[5]   # e5 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)

        self.predict_boundary = predict_boundary        
       
        # Boundary enhancement
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        e1, e2, e3, e4, e5 = features[1], features[2], features[3], features[4], features[5]
        
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])  # Same as e4
        size_d3 = (e3.shape[2], e3.shape[3])  # Same as e3
        size_d2 = (e2.shape[2], e2.shape[3])  # Same as e2
        size_d1 = (e1.shape[2], e1.shape[3])  # Same as e1
        
        # Decoder 4: d4 = e1 + e2 + e3 + e4 + e5
        d4 = self.decoder4([e1, e2, e3, e4, e5], size_d4)
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        d3 = self.decoder3([e1, e2, e3, d4, e5], size_d3)
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        d2 = self.decoder2([e1, e2, d3, d4, e5], size_d2)
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        d1 = self.decoder1([e1, d2, d3, d4, e5], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)

        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(d1, return_boundary=True)
        else:
            decoder_output = self.bem(d1, return_boundary=False)
            
        # Segmentation
        mask = self.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask

class UNet3Plus_B3_CBAM(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder + CBAM on stages 4 & 5 + BEM
    Full-scale skip connections with attention mechanism and boundary enhancement
    
    Architecture:
    - Encoder: EfficientNet-B3 (depth=5)
      ├─ Stage 1: 40 channels ← CBAM applied here
      ├─ Stage 2: 32 channels ← CBAM applied here
      ├─ Stage 3: 48 channels ← CBAM applied here
      ├─ Stage 4: 136 channels ← CBAM applied here
      └─ Stage 5: 384 channels ← CBAM applied here
    
    - CBAM Mechanism:
      ├─ Channel Attention: Focus on "what" features
      ├─ Spatial Attention: Focus on "where" features
      └─ Applied sequentially to enhance deep features
    
    - Decoder: UNet3+ (4 levels with full-scale skip connections)
    
    Args:
        n_classes: Number of output classes (default: 1)
        encoder: Encoder name (default: "efficientnet-b3")
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3"):
        super().__init__()
        
        # ===================== ENCODER =====================
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # ===================== CBAM MODULES =====================
        # CBAM applied to encoder stages 4 & 5 (deepest levels with semantic features)
        # This enhances important features while suppressing noise
        self.cbam_stage1 = CBAM(
            in_channels=encoder_channels[1],  
            reduction_ratio=16,
            kernel_size=7
        )
        self.cbam_stage2 = CBAM(
            in_channels=encoder_channels[2],   
            reduction_ratio=16,
            kernel_size=7
        )
        self.cbam_stage3 = CBAM(
            in_channels=encoder_channels[3],  
            reduction_ratio=16,
            kernel_size=7
        )
        # Stage 4 (e4): 136 channels, 1/8 resolution
        # Focuses on high-level semantic information
        self.cbam_stage4 = CBAM(
            in_channels=encoder_channels[4],  # 136
            reduction_ratio=16,
            kernel_size=7
        )
        
        # Stage 5 (e5): 384 channels, 1/16 resolution
        # Deepest level with most abstract features
        self.cbam_stage5 = CBAM(
            in_channels=encoder_channels[5],  # 384
            reduction_ratio=16,
            kernel_size=7
        )
        
        # ===================== DECODER =====================
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                encoder_channels[3],  # e3: 48
                encoder_channels[4],  # e4: 136
                encoder_channels[5]   # e5: 384
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                encoder_channels[3],  # e3: 48
                decoder_channels,     # d4: 64 (NOT e4!)
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                decoder_channels,     # d3: 64 (NOT e3!)
                decoder_channels,     # d4: 64
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                decoder_channels,     # d2: 64 (NOT e2!)
                decoder_channels,     # d3: 64
                decoder_channels,     # d4: 64
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # ===================== HEADS =====================
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
    
    def forward(self, x, return_boundary=False):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
            return_boundary: Whether to return boundary prediction (default: False)
        
        Returns:
            If return_boundary=False:
                mask: Segmentation output (B, 1, H, W)
            If return_boundary=True:
                (mask, boundary): Segmentation and boundary outputs
        """
        input_size = x.shape[2:]
        
        # ===================== ENCODER FORWARD =====================
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        
        # Apply CBAM to stages 4 & 5 (deep semantic features)
        # This enhances important features and suppresses noise
        #e1 = features[1]                    # e1: 40 channels (no CBAM)
        #e2 = features[2]                    # e2: 32 channels (no CBAM)
        #e3 = features[3]                    # e3: 48 channels (no CBAM)
        e1 = self.cbam_stage1(features[1])  # e1: 40 channels → CBAM
        e2 = self.cbam_stage2(features[2])  # e2: 32 channels → CBAM
        e3 = self.cbam_stage3(features[3])  # e3: 48 channels → CBAM
        e4 = self.cbam_stage4(features[4])  # e4: 136 channels → CBAM
        e5 = self.cbam_stage5(features[5])  # e5: 384 channels → CBAM
        
        # ===================== DECODER FORWARD =====================
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])  # 1/8
        size_d3 = (e3.shape[2], e3.shape[3])  # 1/4
        size_d2 = (e2.shape[2], e2.shape[3])  # 1/2
        size_d1 = (e1.shape[2], e1.shape[3])  # 1/1
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        d4 = self.decoder4([e1, e2, e3, e4, e5], size_d4)
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        d3 = self.decoder3([e1, e2, e3, d4, e5], size_d3)
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        d2 = self.decoder2([e1, e2, d3, d4, e5], size_d2)
        
        # Decoder 1 (shallowest): d1 = e1 + d2 + d3 + d4 + e5
        d1 = self.decoder1([e1, d2, d3, d4, e5], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)      

        # Segmentation head
        mask = self.segmentation_head(d1)
        return mask
            
class UNet3Plus_B3_BEM_CBAM(nn.Module):
    """
    UNet3+ with EfficientNet-B3 encoder + CBAM on stages 4 & 5 + BEM
    Full-scale skip connections with attention mechanism and boundary enhancement
    
    Architecture:
    - Encoder: EfficientNet-B3 (depth=5)
      ├─ Stage 1: 40 channels ← CBAM applied here
      ├─ Stage 2: 32 channels ← CBAM applied here
      ├─ Stage 3: 48 channels ← CBAM applied here
      ├─ Stage 4: 136 channels ← CBAM applied here
      └─ Stage 5: 384 channels ← CBAM applied here
    
    - CBAM Mechanism:
      ├─ Channel Attention: Focus on "what" features
      ├─ Spatial Attention: Focus on "where" features
      └─ Applied sequentially to enhance deep features
    
    - Decoder: UNet3+ (4 levels with full-scale skip connections)
    
    - BEM: Boundary Enhancement Module
      └─ Optional boundary prediction for improved boundary quality
    Args:
        n_classes: Number of output classes (default: 1)
        encoder: Encoder name (default: "efficientnet-b3")
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__()
        
        # ===================== ENCODER =====================
        # EfficientNet-B3 encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Encoder output channels: [3, 40, 32, 48, 136, 384]
        encoder_channels = self.encoder.out_channels
        
        # ===================== CBAM MODULES =====================
        # CBAM applied to encoder stages 4 & 5 (deepest levels with semantic features)
        # This enhances important features while suppressing noise
        self.cbam_stage1 = CBAM(
            in_channels=encoder_channels[1],  
            reduction_ratio=16,
            kernel_size=7
        )
        self.cbam_stage2 = CBAM(
            in_channels=encoder_channels[2],   
            reduction_ratio=16,
            kernel_size=7
        )
        self.cbam_stage3 = CBAM(
            in_channels=encoder_channels[3],  
            reduction_ratio=16,
            kernel_size=7
        )
        # Stage 4 (e4): 136 channels, 1/8 resolution
        # Focuses on high-level semantic information
        self.cbam_stage4 = CBAM(
            in_channels=encoder_channels[4],  # 136
            reduction_ratio=16,
            kernel_size=7
        )
        
        # Stage 5 (e5): 384 channels, 1/16 resolution
        # Deepest level with most abstract features
        self.cbam_stage5 = CBAM(
            in_channels=encoder_channels[5],  # 384
            reduction_ratio=16,
            kernel_size=7
        )
        
        # ===================== DECODER =====================
        # Decoder channels (progressively decrease)
        decoder_channels = 64
        
        # UNet3+ decoder blocks (4 decoder levels)
        # Each block aggregates features from ALL encoder levels
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                encoder_channels[3],  # e3: 48
                encoder_channels[4],  # e4: 136
                encoder_channels[5]   # e5: 384
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                encoder_channels[3],  # e3: 48
                decoder_channels,     # d4: 64 (NOT e4!)
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                encoder_channels[2],  # e2: 32
                decoder_channels,     # d3: 64 (NOT e3!)
                decoder_channels,     # d4: 64
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # Decoder 1: d1 = e1 + d2 + d3 + d4 + e5
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 40
                decoder_channels,     # d2: 64 (NOT e2!)
                decoder_channels,     # d3: 64
                decoder_channels,     # d4: 64
                encoder_channels[5]   # e5: 384 (always present!)
            ],
            out_channels=decoder_channels
        )
        
        # ===================== HEADS =====================
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module (optional)
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
    
    def forward(self, x, return_boundary=False):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
            return_boundary: Whether to return boundary prediction (default: False)
        
        Returns:
            If return_boundary=False:
                mask: Segmentation output (B, 1, H, W)
            If return_boundary=True:
                (mask, boundary): Segmentation and boundary outputs
        """
        input_size = x.shape[2:]
        
        # ===================== ENCODER FORWARD =====================
        features = self.encoder(x)  # [e0, e1, e2, e3, e4, e5]
        
        # Apply CBAM to stages 4 & 5 (deep semantic features)
        # This enhances important features and suppresses noise
        #e1 = features[1]                    # e1: 40 channels (no CBAM)
        #e2 = features[2]                    # e2: 32 channels (no CBAM)
        #e3 = features[3]                    # e3: 48 channels (no CBAM)
        e1 = self.cbam_stage1(features[1])  # e1: 40 channels → CBAM
        e2 = self.cbam_stage2(features[2])  # e2: 32 channels → CBAM
        e3 = self.cbam_stage3(features[3])  # e3: 48 channels → CBAM
        e4 = self.cbam_stage4(features[4])  # e4: 136 channels → CBAM
        e5 = self.cbam_stage5(features[5])  # e5: 384 channels → CBAM
        
        # ===================== DECODER FORWARD =====================
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])  # 1/8
        size_d3 = (e3.shape[2], e3.shape[3])  # 1/4
        size_d2 = (e2.shape[2], e2.shape[3])  # 1/2
        size_d1 = (e1.shape[2], e1.shape[3])  # 1/1
        
        # Decoder 4 (deepest): d4 = e1 + e2 + e3 + e4 + e5
        d4 = self.decoder4([e1, e2, e3, e4, e5], size_d4)
        
        # Decoder 3: d3 = e1 + e2 + e3 + d4 + e5
        d3 = self.decoder3([e1, e2, e3, d4, e5], size_d3)
        
        # Decoder 2: d2 = e1 + e2 + d3 + d4 + e5
        d2 = self.decoder2([e1, e2, d3, d4, e5], size_d2)
        
        # Decoder 1 (shallowest): d1 = e1 + d2 + d3 + d4 + e5
        d1 = self.decoder1([e1, d2, d3, d4, e5], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # ===================== BEM & HEADS =====================
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(d1, return_boundary=True)
        else:
            decoder_output = self.bem(d1, return_boundary=False)
        
        # Segmentation head
        mask = self.segmentation_head(decoder_output)
        
        # Return
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask