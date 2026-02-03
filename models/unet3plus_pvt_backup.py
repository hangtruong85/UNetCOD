import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .boundary_enhancement import BoundaryEnhancementModule
from .cbam import CBAM


# ============================================================================
# UNet3+ Decoder Block
# ============================================================================
"""
UNet3+ Decoder Block for full-scale skip connections
Aggregates features from ALL encoder levels
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

class UNet3Plus_PVT(nn.Module):
    """
    UNet3+ with PVT-V2-B2 encoder
    Full-scale skip connections
    
    Args:
        n_classes: Number of output classes (default: 1)
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Calculate target sizes
        size_d4 = (features[3].shape[2], features[3].shape[3])
        size_d3 = (features[2].shape[2], features[2].shape[3])
        size_d2 = (features[1].shape[2], features[1].shape[3])
        size_d1 = (features[0].shape[2], features[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features[0], features[1], features[2], features[3]], size_d4)
        d3 = self.decoder3([features[0], features[1], features[2], d4], size_d3)
        d2 = self.decoder2([features[0], features[1], d3, d4], size_d2)
        d1 = self.decoder1([features[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask


class UNet3Plus_PVT_BEM(nn.Module):
    """
    UNet3+ with PVT-V2-B2 encoder + Boundary Enhancement Module
    Full-scale skip connections with boundary prediction
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Calculate target sizes
        size_d4 = (features[3].shape[2], features[3].shape[3])
        size_d3 = (features[2].shape[2], features[2].shape[3])
        size_d2 = (features[1].shape[2], features[1].shape[3])
        size_d1 = (features[0].shape[2], features[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features[0], features[1], features[2], features[3]], size_d4)
        d3 = self.decoder3([features[0], features[1], features[2], d4], size_d3)
        d2 = self.decoder2([features[0], features[1], d3, d4], size_d2)
        d1 = self.decoder1([features[0], d2, d3, d4], size_d1)
        
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


class UNet3Plus_PVT_CBAM(nn.Module):
    """
    UNet3+ with PVT-V2-B2 encoder + CBAM on encoder levels 3 & 4
    Full-scale skip connections with channel and spatial attention
    
    Args:
        n_classes: Number of output classes (default: 1)
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # CBAM modules for encoder levels 3 & 4
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        features_with_cbam = [
            features[0],                    # e1 (no CBAM)
            features[1],                    # e2 (no CBAM)
            self.cbam_e3(features[2]),      # e3 (with CBAM)
            self.cbam_e4(features[3])       # e4 (with CBAM)
        ]
        
        # Calculate target sizes
        size_d4 = (features_with_cbam[3].shape[2], features_with_cbam[3].shape[3])
        size_d3 = (features_with_cbam[2].shape[2], features_with_cbam[2].shape[3])
        size_d2 = (features_with_cbam[1].shape[2], features_with_cbam[1].shape[3])
        size_d1 = (features_with_cbam[0].shape[2], features_with_cbam[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], features_with_cbam[3]], size_d4)
        d3 = self.decoder3([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], d4], size_d3)
        d2 = self.decoder2([features_with_cbam[0], features_with_cbam[1], d3, d4], size_d2)
        d1 = self.decoder1([features_with_cbam[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask


class UNet3Plus_PVT_BEM_CBAM(nn.Module):
    """
    UNet3+ with PVT-V2-B2 encoder + BEM + CBAM on encoder levels 3 & 4
    Full-scale skip connections with boundary prediction and attention
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # CBAM modules for encoder levels 3 & 4
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = UNet3PlusDecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        features_with_cbam = [
            features[0],                    # e1 (no CBAM)
            features[1],                    # e2 (no CBAM)
            self.cbam_e3(features[2]),      # e3 (with CBAM)
            self.cbam_e4(features[3])       # e4 (with CBAM)
        ]
        
        # Calculate target sizes
        size_d4 = (features_with_cbam[3].shape[2], features_with_cbam[3].shape[3])
        size_d3 = (features_with_cbam[2].shape[2], features_with_cbam[2].shape[3])
        size_d2 = (features_with_cbam[1].shape[2], features_with_cbam[1].shape[3])
        size_d1 = (features_with_cbam[0].shape[2], features_with_cbam[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], features_with_cbam[3]], size_d4)
        d3 = self.decoder3([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], d4], size_d3)
        d2 = self.decoder2([features_with_cbam[0], features_with_cbam[1], d3, d4], size_d2)
        d1 = self.decoder1([features_with_cbam[0], d2, d3, d4], size_d1)
        
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