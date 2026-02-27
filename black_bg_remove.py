import torch
import numpy as np
from scipy import ndimage

class BlackBGRemoveByDistance:
    """
    Remove (make transparent) pixels close to pure black using RGB distance.
    distance = sqrt(r^2 + g^2 + b^2)
    If distance < threshold => alpha = 0
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # threshold in 0..1 space (ComfyUI IMAGE is float32 0..1)
                "threshold": ("FLOAT", {
                    "default": 0.06,   # ~15/255
                    "min": 0.0,
                    "max": 1.732,      # sqrt(1^2+1^2+1^2)
                    "step": 0.001,
                    "tooltip": "RGB distance to black (0..~1.732). Smaller = stricter. 0.06≈15/255"
                }),
                "keep_original_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If input has alpha, preserve it for non-removed pixels."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)
    FUNCTION = "remove_black_bg"
    CATEGORY = "image/alpha"

    def remove_black_bg(self, image: torch.Tensor, threshold: float, keep_original_alpha: bool):
        """
        image: ComfyUI IMAGE tensor, typically [B,H,W,3] float 0..1
               sometimes can be [B,H,W,4] if already RGBA
        returns RGBA [B,H,W,4]
        """
        if image.dtype != torch.float32 and image.dtype != torch.float16 and image.dtype != torch.bfloat16:
            image = image.float()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        b, h, w, c = image.shape
        if c not in (3, 4):
            raise ValueError(f"Expected IMAGE with 3 or 4 channels, got {c}")

        rgb = image[..., :3].clamp(0.0, 1.0)

        # distance to black in 0..1 domain
        # sqrt(r^2+g^2+b^2)
        dist = torch.sqrt(torch.sum(rgb * rgb, dim=-1))  # [B,H,W]

        # mask for "near black" pixels
        remove_mask = dist < float(threshold)  # [B,H,W] bool

        if c == 4 and keep_original_alpha:
            alpha = image[..., 3].clamp(0.0, 1.0)
        else:
            alpha = torch.ones((b, h, w), device=image.device, dtype=image.dtype)

        # set alpha=0 where remove_mask is true
        alpha = torch.where(remove_mask, torch.zeros_like(alpha), alpha)

        out = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)  # [B,H,W,4]
        return (out,)


class BlackBGRemoveSmart:
    """
    Remove black background while preserving dark regions inside the subject.
    Only removes black pixel regions that are connected to the image border,
    keeping isolated dark areas (windows, shadows, etc.) intact.
    Optionally applies feathered (smooth) edges at the boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.06,
                    "min": 0.0,
                    "max": 1.732,
                    "step": 0.001,
                    "tooltip": "RGB distance to black (0..~1.732). Smaller = stricter. 0.06≈15/255"
                }),
                "feather_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Gaussian blur sigma for soft edges. 0 = hard edges."
                }),
                "connectivity": (["8-way (diagonal)", "4-way (cross)"], {
                    "default": "8-way (diagonal)",
                    "tooltip": "Pixel connectivity for region detection. 8-way includes diagonals."
                }),
                "keep_original_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If input has alpha, preserve it for non-removed pixels."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)
    FUNCTION = "remove_black_bg_smart"
    CATEGORY = "image/alpha"

    def remove_black_bg_smart(
        self,
        image: torch.Tensor,
        threshold: float,
        feather_radius: int,
        connectivity: str,
        keep_original_alpha: bool,
    ):
        if image.dtype != torch.float32 and image.dtype != torch.float16 and image.dtype != torch.bfloat16:
            image = image.float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        b, h, w, c = image.shape
        if c not in (3, 4):
            raise ValueError(f"Expected IMAGE with 3 or 4 channels, got {c}")

        rgb = image[..., :3].clamp(0.0, 1.0)

        use_8way = connectivity.startswith("8")
        conn = 2 if use_8way else 1
        structure = ndimage.generate_binary_structure(2, conn)

        results = []
        for i in range(b):
            rgb_np = rgb[i].cpu().numpy()  # [H, W, 3]

            dist = np.sqrt(np.sum(rgb_np * rgb_np, axis=-1))  # [H, W]
            black_mask = dist < float(threshold)

            labeled, _ = ndimage.label(black_mask, structure=structure)

            border_labels = set()
            border_labels |= set(labeled[0, :].ravel())
            border_labels |= set(labeled[-1, :].ravel())
            border_labels |= set(labeled[:, 0].ravel())
            border_labels |= set(labeled[:, -1].ravel())
            border_labels.discard(0)

            remove_mask = np.isin(labeled, list(border_labels)) if border_labels else np.zeros_like(black_mask)

            # Dilate remove_mask to capture dark fringe pixels on diagonal edges
            expanded = ndimage.binary_dilation(remove_mask, structure=structure, iterations=2)

            # Use actual color distance for natural anti-aliased alpha in the transition zone.
            # Pixels deep in the background (low dist) → alpha≈0,
            # pixels at the boundary (dist near threshold) → smooth ramp,
            # pixels outside the expanded zone → alpha=1.
            color_alpha = np.clip(dist / float(threshold), 0.0, 1.0).astype(np.float32)
            alpha_np = np.where(expanded, color_alpha, 1.0)

            if feather_radius > 0:
                alpha_np = ndimage.gaussian_filter(alpha_np, sigma=feather_radius)
                alpha_np = np.clip(alpha_np, 0.0, 1.0)

            alpha_t = torch.from_numpy(alpha_np).to(device=image.device, dtype=image.dtype)

            if c == 4 and keep_original_alpha:
                orig_alpha = image[i, ..., 3].clamp(0.0, 1.0)
                alpha_t = torch.min(alpha_t, orig_alpha)

            frame = torch.cat([rgb[i], alpha_t.unsqueeze(-1)], dim=-1)  # [H,W,4]
            results.append(frame)

        out = torch.stack(results, dim=0)  # [B,H,W,4]
        return (out,)

