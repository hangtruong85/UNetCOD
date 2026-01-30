"""
UNet with + BEM for Camouflaged Object Detection
Combines:
- Boundary Enhancement Module (BEM) for weak edge detection
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from models.boundary_enhancement import BoundaryEnhancementModule

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)

# Helper class từ code gốc
class UNet_B3(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)

# Helper class từ code gốc
class UNet_Resnet50(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
    def forward(self, x):
        return self.backbone(x)

class UNet_B3_BEM(nn.Module):
    """
    UNet with BEM 
    Tests the contribution of attention + boundary enhancement
    """
    def __init__(self, n_classes=1, encoder="efficientnet-b3", predict_boundary=True):
        super().__init__()
        
        self.backbone = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            classes=n_classes,
            activation=None,
        )
        
        # Get encoder channels dynamically
        encoder_channels = self.backbone.encoder.out_channels
        
        self.predict_boundary = predict_boundary        
       
        # Boundary enhancement
        self.bem = BoundaryEnhancementModule(16, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        # Encoder
        features = self.backbone.encoder(x)

        # Decoder
        decoder_output = self.backbone.decoder(features)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(decoder_output, return_boundary=True)
        else:
            decoder_output = self.bem(decoder_output, return_boundary=False)
        
        # Segmentation head
        mask = self.backbone.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask
   