import torch
import numpy as np
from scipy import ndimage


def _parse_hex_color(hex_str: str) -> tuple[float, float, float]:
    """Parse hex color string (e.g. '#FF00AA' or 'FF00AA') to (r, g, b) in 0..1 range."""
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: '#{hex_str}'. Expected 6-digit hex like '#000000'.")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def _border_median_color(rgb_np: np.ndarray) -> np.ndarray:
    """Median color of the 1px border ring, used as the actual background color."""
    ring = np.concatenate([
        rgb_np[0, :, :],
        rgb_np[-1, :, :],
        rgb_np[:, 0, :],
        rgb_np[:, -1, :],
    ], axis=0)
    return np.median(ring, axis=0).astype(np.float32)


class RemoveColorBGAdvanced:
    """
    Remove a target-color background while preserving similar-colored regions inside the subject.

    Advanced variant of RemoveColorBG. The detection tolerance and the alpha ramp are
    separate parameters, so background pixels that merely approximate the target color
    (e.g. #FCFEFC against a #FFFFFF target) become fully transparent instead of keeping
    a residual alpha that shows up as noise when composited.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Hex color code of the background to remove (e.g. #000000 for black, #FFFFFF for white, #00FF00 for green)."
                }),
                "auto_detect_color": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Ignore target_color and use the median color of the image border instead. Useful for AI images whose background is near-but-not-exactly the intended color."
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.06,
                    "min": 0.0,
                    "max": 1.732,
                    "step": 0.001,
                    "tooltip": "Colors within this distance of the target are FULLY transparent (alpha 0). 0.06≈15/255."
                }),
                "softness": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.732,
                    "step": 0.001,
                    "tooltip": "Distance range beyond tolerance over which alpha ramps 0->1. 0 = hard cut."
                }),
                "distance_metric": (["euclidean", "max channel"], {
                    "default": "euclidean",
                    "tooltip": "How color distance is measured. 'max channel' is stricter about single-channel tints."
                }),
                "connectivity": (["8-way (diagonal)", "4-way (cross)"], {
                    "default": "8-way (diagonal)",
                    "tooltip": "Pixel connectivity for region detection. 8-way includes diagonals."
                }),
                "edge_expand": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Dilate the detected background region by N pixels so anti-aliased edges are treated as background too."
                }),
                "feather_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Gaussian blur sigma for soft edges. 0 = hard edges."
                }),
                "alpha_clip_low": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Alpha below this is snapped to 0. Kills faint background residue."
                }),
                "alpha_clip_high": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Alpha above this is snapped to 1. Kills faint holes inside the subject."
                }),
                "despill": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unpremultiply the target color out of semi-transparent edge pixels, removing color fringing."
                }),
                "clear_removed_rgb": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set RGB to black where alpha is 0. Only needed if a downstream node ignores alpha."
                }),
                "keep_original_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If input has alpha, preserve it for non-removed pixels."
                }),
            },
            "optional": {
                "onoff": ("BOOLEAN", {
                    "default": True,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "On = remove background. Off = bypass input to output. Missing input defaults to on.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_rgba", "alpha_mask")
    FUNCTION = "remove_color_bg_advanced"
    CATEGORY = "image/alpha"

    def remove_color_bg_advanced(
        self,
        image: torch.Tensor,
        target_color: str,
        auto_detect_color: bool,
        tolerance: float,
        softness: float,
        distance_metric: str,
        connectivity: str,
        edge_expand: int,
        feather_radius: float,
        alpha_clip_low: float,
        alpha_clip_high: float,
        despill: bool,
        clear_removed_rgb: bool,
        keep_original_alpha: bool,
        onoff: bool = True,
    ):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if not onoff:
            mask = torch.ones(image.shape[0], image.shape[1], image.shape[2],
                              device=image.device, dtype=torch.float32)
            return (image, mask)

        if image.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            image = image.float()

        b, h, w, c = image.shape
        if c not in (3, 4):
            raise ValueError(f"Expected IMAGE with 3 or 4 channels, got {c}")

        hex_target = np.array(_parse_hex_color(target_color), dtype=np.float32)

        rgb = image[..., :3].clamp(0.0, 1.0)

        conn = 2 if connectivity.startswith("8") else 1
        structure = ndimage.generate_binary_structure(2, conn)

        tolerance = float(tolerance)
        softness = float(softness)
        ramp = max(softness, 1e-6)
        detect_range = tolerance + softness

        low = min(float(alpha_clip_low), float(alpha_clip_high))
        high = max(float(alpha_clip_low), float(alpha_clip_high))
        levels_span = max(high - low, 1e-6)

        results = []
        masks = []
        for i in range(b):
            rgb_np = rgb[i].cpu().numpy().astype(np.float32)  # [H, W, 3]

            target_np = _border_median_color(rgb_np) if auto_detect_color else hex_target

            diff = np.abs(rgb_np - target_np)
            if distance_metric.startswith("max"):
                dist = np.max(diff, axis=-1)
            else:
                dist = np.sqrt(np.sum(diff * diff, axis=-1))

            color_mask = dist <= detect_range

            labeled, _ = ndimage.label(color_mask, structure=structure)

            border_labels = set()
            border_labels |= set(labeled[0, :].ravel())
            border_labels |= set(labeled[-1, :].ravel())
            border_labels |= set(labeled[:, 0].ravel())
            border_labels |= set(labeled[:, -1].ravel())
            border_labels.discard(0)

            if border_labels:
                remove_mask = np.isin(labeled, list(border_labels))
            else:
                remove_mask = np.zeros_like(color_mask)

            if edge_expand > 0:
                remove_mask = ndimage.binary_dilation(
                    remove_mask, structure=structure, iterations=int(edge_expand)
                )

            color_alpha = np.clip((dist - tolerance) / ramp, 0.0, 1.0).astype(np.float32)
            alpha_np = np.where(remove_mask, color_alpha, 1.0).astype(np.float32)

            alpha_np = np.clip((alpha_np - low) / levels_span, 0.0, 1.0)

            if feather_radius > 0:
                alpha_np = ndimage.gaussian_filter(alpha_np, sigma=float(feather_radius))
                alpha_np = np.clip(alpha_np, 0.0, 1.0)

            out_rgb = rgb_np
            if despill:
                out_rgb = out_rgb.copy()
                edge = (alpha_np > 1e-3) & (alpha_np < 1.0)
                if np.any(edge):
                    a = alpha_np[edge][:, None]
                    observed = out_rgb[edge]
                    unpremultiplied = (observed - (1.0 - a) * target_np[None, :]) / a
                    out_rgb[edge] = np.clip(unpremultiplied, 0.0, 1.0)

            if clear_removed_rgb:
                out_rgb = np.where(alpha_np[..., None] > 0.0, out_rgb, 0.0).astype(np.float32)

            rgb_t = torch.from_numpy(np.ascontiguousarray(out_rgb)).to(
                device=image.device, dtype=image.dtype
            )
            alpha_t = torch.from_numpy(alpha_np).to(device=image.device, dtype=image.dtype)

            if c == 4 and keep_original_alpha:
                orig_alpha = image[i, ..., 3].clamp(0.0, 1.0)
                alpha_t = torch.min(alpha_t, orig_alpha)

            results.append(torch.cat([rgb_t, alpha_t.unsqueeze(-1)], dim=-1))  # [H,W,4]
            masks.append(alpha_t.float())

        out = torch.stack(results, dim=0)  # [B,H,W,4]
        out_mask = torch.stack(masks, dim=0)  # [B,H,W]
        return (out, out_mask)
