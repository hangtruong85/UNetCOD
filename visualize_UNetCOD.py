"""
Visualization tool for UNet3Plus_B3_BEM_CBAM model (VERTICAL / column-major layout)
Layout: 5 rows × 6 columns
    Col 0: Predictions (Input, GT, Prediction, Overlay, Error Map)
    Col 1: Encoder features (e1 .. e5)
    Col 2: CBAM Attention (spatial attention stages 1-5)
    Col 3: Decoder features (d4, d3, d2, d1, BEM Fusion)
    Col 4: BEM analysis (edge features, Sobel, boundary pred, overlay, contours)
    Col 5: Attention maps / CAM (d4, d3, d2, d1, BEM)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from datasets.mhcd_dataset import MHCDDataset
from models.unet3plus import UNet3Plus_B3_BEM_CBAM


# ============================================================================
# Hook-based Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract intermediate features from model layers using forward hooks."""

    def __init__(self):
        self.features = {}
        self.hooks = []

    def register_hook(self, module, name):
        hook = module.register_forward_hook(self._make_hook(name))
        self.hooks.append(hook)

    def register_hooks_by_path(self, model, layer_paths):
        for name, path in layer_paths.items():
            module = model
            for attr in path.split('.'):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    print(f"  Warning: cannot find '{attr}' in path '{path}'")
                    module = None
                    break
            if module is not None:
                self.register_hook(module, name)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                self.features[name] = output[0].detach()
        return hook_fn

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================================
# Attention Map Utilities
# ============================================================================

def generate_cam(features, method='norm'):
    if method == 'mean':
        return features.mean(dim=1, keepdim=True)
    elif method == 'max':
        return features.max(dim=1, keepdim=True)[0]
    elif method == 'norm':
        return torch.norm(features, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown CAM method: {method}")


def normalize_to_numpy(tensor):
    t = tensor.squeeze().cpu().float()
    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)
    return t.numpy()


def tensor_to_rgb(image_tensor):
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# ============================================================================
# Font sizes (centralized for easy tuning)
# ============================================================================
TITLE_FONTSIZE = 12          # cell titles
COL_HEADER_FONTSIZE = 14     # column headers at top
SUPTITLE_FONTSIZE = 18       # figure super title
COLORBAR_FONTSIZE = 9        # colorbar tick labels
NOTFOUND_FONTSIZE = 11       # "not found" placeholder text


# ============================================================================
# UNet3Plus_B3_BEM_CBAM Visualizer  (column-major layout)
# ============================================================================

class UNet3PlusCBAMVisualizer:
    """
    Column-major visualization for UNet3Plus_B3_BEM_CBAM.

    Grid: 5 rows × 6 columns
        Col 0 - Predictions:   Input | GT | Prediction | Overlay | Error Map
        Col 1 - Encoder:       e1 | e2 | e3 | e4 | e5
        Col 2 - CBAM:          S1 | S2 | S3 | S4 | S5  (spatial attention)
        Col 3 - Decoder+BEM:   d4 | d3 | d2 | d1 | BEM Fusion
        Col 4 - BEM Analysis:  Edge feat | Sobel GT | Boundary pred | Boundary overlay | Contours
        Col 5 - CAM:           d4 | d3 | d2 | d1 | BEM
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_encoder_hooks(self):
        encoder = self.model.encoder
        def encoder_hook(module, input, output):
            if isinstance(output, (list, tuple)):
                for i, feat in enumerate(output):
                    if isinstance(feat, torch.Tensor):
                        self.extractor.features[f'e{i}'] = feat.detach()
        hook = encoder.register_forward_hook(encoder_hook)
        self.extractor.hooks.append(hook)

    def _register_cbam_hooks(self):
        for stage_idx in range(1, 6):
            cbam_module = getattr(self.model, f'cbam_stage{stage_idx}', None)
            if cbam_module is None:
                continue
            self.extractor.register_hook(cbam_module.channel_attention, f'cbam_s{stage_idx}_channel')
            self.extractor.register_hook(cbam_module.spatial_attention, f'cbam_s{stage_idx}_spatial')
            self.extractor.register_hook(cbam_module, f'cbam_s{stage_idx}_output')

    def _register_decoder_hooks(self):
        for name in ['decoder4', 'decoder3', 'decoder2', 'decoder1']:
            module = getattr(self.model, name, None)
            if module is not None:
                self.extractor.register_hook(module, name.replace('decoder', 'd'))

    def _register_bem_hooks(self):
        bem = self.model.bem
        self.extractor.register_hook(bem.edge_conv, 'bem_edge_conv')
        self.extractor.register_hook(bem.fusion, 'bem_fusion')
        if hasattr(bem, 'boundary_head'):
            self.extractor.register_hook(bem.boundary_head, 'bem_boundary_head')

    # ------------------------------------------------------------------
    # Forward with hooks
    # ------------------------------------------------------------------

    def _forward_with_hooks(self, image):
        self.extractor.clear()
        self.extractor.remove_hooks()
        self._register_encoder_hooks()
        self._register_cbam_hooks()
        self._register_decoder_hooks()
        self._register_bem_hooks()

        with torch.no_grad():
            output = self.model(image, return_boundary=True)

        if isinstance(output, tuple):
            pred_logit, boundary_logit = output
        else:
            pred_logit, boundary_logit = output, None
        return pred_logit, boundary_logit

    # ------------------------------------------------------------------
    # Helper: add colorbar with controlled font
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cbar(im, ax):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=COLORBAR_FONTSIZE)

    @staticmethod
    def _not_found(ax, label):
        ax.text(0.5, 0.5, f'{label}\nnot found',
                ha='center', va='center', fontsize=NOTFOUND_FONTSIZE)

    # ------------------------------------------------------------------
    # Column renderers  (each fills axes[row] for row 0..4)
    # ------------------------------------------------------------------

    def _draw_col_predictions(self, axes, img_np, mask_np, pred_np):
        """Col 0: Input, GT, Prediction, Overlay, Error Map  (top→bottom)."""
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image', fontsize=TITLE_FONTSIZE, fontweight='bold')

        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth', fontsize=TITLE_FONTSIZE, fontweight='bold')

        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title('Prediction', fontsize=TITLE_FONTSIZE, fontweight='bold')

        axes[3].imshow(img_np)
        axes[3].imshow(pred_np, cmap='jet', alpha=0.5)
        axes[3].set_title('Overlay', fontsize=TITLE_FONTSIZE, fontweight='bold')

        error = np.abs(mask_np - pred_np)
        im = axes[4].imshow(error, cmap='hot', vmin=0, vmax=1)
        axes[4].set_title(f'Error (MAE={error.mean():.4f})', fontsize=TITLE_FONTSIZE, fontweight='bold')
        self._add_cbar(im, axes[4])

    def _draw_col_encoder(self, axes):
        """Col 1: Encoder features e1..e5  (top→bottom)."""
        features = self.extractor.features
        for i in range(5):
            key = f'e{i+1}'
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method='norm')
                cam_np = normalize_to_numpy(cam[0])
                im = ax.imshow(cam_np, cmap='jet')
                ax.set_title(f'{key}  ({feat.shape[1]}ch, {feat.shape[2]}×{feat.shape[3]})',
                             fontsize=TITLE_FONTSIZE)
                self._add_cbar(im, ax)
            else:
                self._not_found(ax, key)

    def _draw_col_cbam(self, axes, image):
        """Col 2: CBAM spatial attention for stages 1-5  (top→bottom)."""
        features = self.extractor.features
        img_size = image.shape[2:]
        img_np = tensor_to_rgb(image[0])

        for i in range(5):
            stage = i + 1
            ax = axes[i]
            spatial_key = f'cbam_s{stage}_spatial'
            channel_key = f'cbam_s{stage}_channel'

            if spatial_key in features:
                spatial_att = features[spatial_key]
                spatial_up = F.interpolate(spatial_att, size=img_size,
                                           mode='bilinear', align_corners=False)
                spatial_np = normalize_to_numpy(spatial_up[0])

                ax.imshow(img_np)
                im = ax.imshow(spatial_np, cmap='jet', alpha=0.55)

                ch_info = ''
                if channel_key in features:
                    ch_att = features[channel_key]
                    ch_info = f'  Ch: μ={ch_att.mean().item():.3f}'

                ax.set_title(
                    f'CBAM S{stage} ({spatial_att.shape[2]}×{spatial_att.shape[3]}){ch_info}',
                    fontsize=TITLE_FONTSIZE)
                self._add_cbar(im, ax)
            else:
                self._not_found(ax, f'CBAM S{stage}')

    def _draw_col_decoder(self, axes):
        """Col 3: Decoder d4,d3,d2,d1 + BEM Fusion  (top→bottom)."""
        features = self.extractor.features
        keys = ['d4', 'd3', 'd2', 'd1']

        for i, key in enumerate(keys):
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method='norm')
                cam_np = normalize_to_numpy(cam[0])
                im = ax.imshow(cam_np, cmap='jet')
                ax.set_title(f'{key}  ({feat.shape[1]}ch, {feat.shape[2]}×{feat.shape[3]})',
                             fontsize=TITLE_FONTSIZE)
                self._add_cbar(im, ax)
            else:
                self._not_found(ax, key)

        # Row 4: BEM fusion
        ax = axes[4]
        if 'bem_fusion' in features:
            feat = features['bem_fusion']
            cam = generate_cam(feat, method='norm')
            cam_np = normalize_to_numpy(cam[0])
            im = ax.imshow(cam_np, cmap='jet')
            ax.set_title(f'BEM Fusion ({feat.shape[1]}ch)', fontsize=TITLE_FONTSIZE, fontweight='bold')
            self._add_cbar(im, ax)
        else:
            self._not_found(ax, 'BEM Fusion')

    def _draw_col_bem(self, axes, img_np, mask_np, pred_np, boundary_logit):
        """Col 4: BEM analysis  (top→bottom)."""
        features = self.extractor.features

        # Row 0: BEM edge conv features
        ax = axes[0]
        if 'bem_edge_conv' in features:
            feat = features['bem_edge_conv']
            cam = generate_cam(feat, method='norm')
            cam_np = normalize_to_numpy(cam[0])
            im = ax.imshow(cam_np, cmap='inferno')
            ax.set_title('BEM Edge Features', fontsize=TITLE_FONTSIZE, fontweight='bold')
            self._add_cbar(im, ax)
        else:
            self._not_found(ax, 'Edge features')

        # Row 1: Sobel boundary from GT
        ax = axes[1]
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        sobel_boundary = self.model.bem.extract_boundary_map(mask_t)
        sobel_np = normalize_to_numpy(sobel_boundary[0])
        im = ax.imshow(sobel_np, cmap='hot')
        ax.set_title('GT Boundary (Sobel)', fontsize=TITLE_FONTSIZE, fontweight='bold')
        self._add_cbar(im, ax)

        # Row 2: Boundary prediction
        ax = axes[2]
        if boundary_logit is not None:
            boundary_pred = torch.sigmoid(boundary_logit)
            boundary_np = boundary_pred[0, 0].cpu().numpy()
            im = ax.imshow(boundary_np, cmap='hot')
            ax.set_title('Boundary Pred', fontsize=TITLE_FONTSIZE, fontweight='bold')
            self._add_cbar(im, ax)
        elif 'bem_boundary_head' in features:
            feat = features['bem_boundary_head']
            boundary_np = torch.sigmoid(feat[0, 0]).cpu().numpy()
            im = ax.imshow(boundary_np, cmap='hot')
            ax.set_title('Boundary Pred', fontsize=TITLE_FONTSIZE, fontweight='bold')
            self._add_cbar(im, ax)
        else:
            self._not_found(ax, 'Boundary pred')

        # Row 3: Boundary overlay on image
        ax = axes[3]
        ax.imshow(img_np)
        if boundary_logit is not None:
            boundary_overlay = torch.sigmoid(boundary_logit)[0, 0].cpu().numpy()
            ax.imshow(boundary_overlay, cmap='Reds', alpha=0.6)
        ax.set_title('Boundary Overlay', fontsize=TITLE_FONTSIZE, fontweight='bold')

        # Row 4: Contours
        ax = axes[4]
        ax.imshow(img_np)
        ax.contour(mask_np, levels=[0.5], colors='lime', linewidths=1.5)
        ax.contour(pred_np, levels=[0.5], colors='red', linewidths=1.5)
        ax.set_title('Contours (G=GT, R=Pred)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    def _draw_col_cam(self, axes, image):
        """Col 5: CAM attention overlays  (top→bottom)."""
        features = self.extractor.features
        methods = [('d4', 'mean'), ('d3', 'mean'), ('d2', 'max'), ('d1', 'max'), ('bem_fusion', 'norm')]
        labels = ['d4 (mean)', 'd3 (mean)', 'd2 (max)', 'd1 (max)', 'BEM (norm)']
        img_size = image.shape[2:]

        for i, ((key, method), label) in enumerate(zip(methods, labels)):
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method=method)
                cam_up = F.interpolate(cam, size=img_size, mode='bilinear', align_corners=False)
                cam_np = normalize_to_numpy(cam_up[0])

                img_np = tensor_to_rgb(image[0])
                ax.imshow(img_np)
                ax.imshow(cam_np, cmap='jet', alpha=0.5)
                ax.set_title(f'CAM: {label}', fontsize=TITLE_FONTSIZE)
            else:
                self._not_found(ax, key)

    # ------------------------------------------------------------------
    # Main visualization entry
    # ------------------------------------------------------------------

    def visualize_sample(self, image, mask, save_path):
        """
        Create full visualization for one sample.
        Args:
            image: (1, 3, H, W) tensor
            mask: (1, 1, H, W) tensor
            save_path: output image path
        """
        image = image.to(self.device)
        mask = mask.to(self.device)

        # Forward pass with hooks
        pred_logit, boundary_logit = self._forward_with_hooks(image)
        pred_prob = torch.sigmoid(pred_logit)

        # Convert to numpy
        img_np = tensor_to_rgb(image[0])
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_prob[0, 0].cpu().numpy()

        # ---- Create 5-row × 6-col figure (column-major) ----
        fig = plt.figure(figsize=(30, 22))
        gs = GridSpec(5, 6, figure=fig, hspace=0.12, wspace=0.10)

        # Build axes[row][col]
        axes = [[fig.add_subplot(gs[r, c]) for c in range(6)] for r in range(5)]

        # Extract column-wise slices: axes_col[col] = [axes[0][col], ..., axes[4][col]]
        def col_axes(c):
            return [axes[r][c] for r in range(5)]

        # Fill columns
        self._draw_col_predictions(col_axes(0), img_np, mask_np, pred_np)
        self._draw_col_encoder(col_axes(1))
        self._draw_col_cbam(col_axes(2), image)
        self._draw_col_decoder(col_axes(3))
        self._draw_col_bem(col_axes(4), img_np, mask_np, pred_np, boundary_logit)
        self._draw_col_cam(col_axes(5), image)

        # Turn off all axes ticks
        for row in axes:
            for ax in row:
                ax.axis('off')

        # Column headers at the top of each column
        col_labels = [
            'Predictions',
            'Encoder (e1→e5)',
            'CBAM Attention',
            'Decoder + BEM',
            'BEM Analysis',
            'CAM Overlay',
        ]
        for c, label in enumerate(col_labels):
            axes[0][c].annotate(
                label, xy=(0.5, 1.15), xycoords='axes fraction',
                fontsize=COL_HEADER_FONTSIZE, fontweight='bold',
                ha='center', va='bottom',
            )

        #fig.suptitle('UNet3Plus_B3_BEM_CBAM  —  Visualization',
        #             fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=1.0)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Cleanup
        self.extractor.clear()
        self.extractor.remove_hooks()

        print(f"  Saved: {save_path}")


# ============================================================================
# Batch Visualization
# ============================================================================

def visualize_batch(model, dataset, indices, save_dir, device):
    vis = UNet3PlusCBAMVisualizer(model, device)
    for idx in indices:
        image, mask = dataset[idx]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        save_path = os.path.join(save_dir, f"sample_{idx:04d}.png")
        print(f"Processing sample {idx} ...")
        vis.visualize_sample(image, mask, save_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize UNet3Plus_B3_BEM_CBAM features and attention")
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--root', type=str, default='../MHCD_seg',
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--img_size', type=int, default=352,
                        help='Input image size')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save output images')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("UNet3Plus_B3_BEM_CBAM  --  Feature & Attention Visualization")
    print("=" * 70)

    # --- Create model ---
    print(f"\nCreating UNet3Plus_B3_BEM_CBAM model ...")
    model = UNet3Plus_B3_BEM_CBAM(n_classes=1, predict_boundary=True)

    # --- Load checkpoint ---
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            epoch = ckpt.get('epoch', '?')
            best = ckpt.get('best_s_measure', '?')
            print(f"  Loaded from epoch {epoch}, best S-measure: {best}")
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
            print(f"  Loaded state_dict")
        else:
            print(f"  Warning: unrecognized checkpoint format")
    else:
        print("  No checkpoint provided -- using random weights (for structure testing)")

    model = model.to(device).eval()

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.root} [{args.split}] ...")
    dataset = MHCDDataset(args.root, args.split, args.img_size)
    print(f"  Total samples: {len(dataset)}")

    # --- Select samples ---
    num = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num, dtype=int)

    # --- Visualize ---
    save_dir = os.path.join(args.save_dir, 'UNet3Plus_B3_BEM_CBAM')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nVisualizing {num} samples -> {save_dir}/\n")

    visualize_batch(model, dataset, indices, save_dir, device)

    print("\n" + "=" * 70)
    print(f"Done. All visualizations saved to: {save_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
  