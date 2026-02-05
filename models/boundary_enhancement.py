"""
Boundary Enhancement Module for Camouflaged Object Detection
Uses Sobel filters to enhance weak boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryEnhancementModule(nn.Module):
    """
    Enhance weak boundaries using edge-aware features
    Uses Sobel filters for boundary detection
    Can optionally predict boundary map for boundary loss
    """
    def __init__(self, channels, predict_boundary=False):
        super().__init__()
        
        self.predict_boundary = predict_boundary
        
        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Sobel filters for boundary detection (fixed, not learned)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3))
        
        # Fusion of original and edge-enhanced features
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
        # Optional boundary prediction head
        if predict_boundary:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, 1)  # Predict boundary map
            )
        
    def forward(self, x, return_boundary=False):
        """
        Args:
            x: input features (B, C, H, W)
            return_boundary: whether to return boundary prediction
        Returns:
            If return_boundary=False: enhanced features only
            If return_boundary=True: (enhanced_features, boundary_pred)
        """
        # Extract edge features
        edge_feat = self.edge_conv(x)
        
        # Combine original and edge-enhanced features
        out = torch.cat([x, edge_feat], dim=1)
        out = self.fusion(out)
        
        # Optionally predict boundary
        if return_boundary and self.predict_boundary:
            boundary_pred = self.boundary_head(out)
            return out, boundary_pred
        else:
            return out
    
    def extract_boundary_map(self, x):
        """
        Explicitly extract boundary map using Sobel filters
        Useful for creating ground truth boundary maps
        Args:
            x: input features (B, C, H, W) or (B, 1, H, W)
        Returns:
            boundary magnitude map (B, 1, H, W)
        """
        # Average across channels if multi-channel
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x
        
        # Apply Sobel filters
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        boundary = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Normalize to [0, 1]
        boundary = boundary / (boundary.max() + 1e-8)  # OR
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary
