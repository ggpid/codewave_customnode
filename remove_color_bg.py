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


def _detect_bg_color(rgb_np: np.ndarray) -> np.ndarray:
    """Estimate background color from corners, falling back to the dominant border color."""
    h, w, _ = rgb_np.shape
    bw = max(1, min(8, h // 8, w // 8))

    corners = [
        rgb_np[:bw, :bw],
        rgb_np[:bw, w - bw :],
        rgb_np[h - bw :, :bw],
        rgb_np[h - bw :, w - bw :],
    ]
    cmed = np.stack([np.median(c.reshape(-1, 3), axis=0) for c in corners], axis=0)
    diffs = np.linalg.norm(cmed[:, None, :] - cmed[None, :, :], axis=-1)
    agree = (diffs < 0.12).sum(axis=1)
    best = int(np.argmax(agree))
    if agree[best] >= 2:
        return cmed[diffs[best] < 0.12].mean(axis=0).astype(np.float32)

    top = rgb_np[:bw].reshape(-1, 3)
    bot = rgb_np[-bw:].reshape(-1, 3)
    mid_h = h > 2 * bw
    left = rgb_np[bw : h - bw, :bw].reshape(-1, 3) if mid_h else np.zeros((0, 3), np.float32)
    right = rgb_np[bw : h - bw, -bw:].reshape(-1, 3) if mid_h else np.zeros((0, 3), np.float32)
    border = np.concatenate([top, bot, left, right], axis=0)

    q = np.clip((border * 31.0).astype(np.int32), 0, 31)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    mode = int(np.argmax(np.bincount(keys, minlength=32768)))
    return np.array(
        [mode // 1024, (mode // 32) % 32, mode % 32],
        dtype=np.float32,
    ) / 31.0


def _apply_despill(rgb_np: np.ndarray, key_color: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reduce key-color spill on masked pixels (typical chroma-key despill)."""
    channels = np.asarray(key_color, dtype=np.float32)
    if float(channels.max() - channels.min()) < 0.08:
        return rgb_np

    key_ch = int(np.argmax(channels))
    others = [i for i in range(3) if i != key_ch]
    out = rgb_np.copy()
    key_vals = out[..., key_ch]
    other_avg = (out[..., others[0]] + out[..., others[1]]) * 0.5
    spill = np.maximum(key_vals - other_avg, 0.0) * mask.astype(np.float32)

    out[..., key_ch] = key_vals - spill
    restore = spill * 0.25
    out[..., others[0]] = np.clip(out[..., others[0]] + restore, 0.0, 1.0)
    out[..., others[1]] = np.clip(out[..., others[1]] + restore, 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


class RemoveColorBG:
    """
    Remove a target-color background while preserving similar-colored regions inside the subject.
    Only removes pixel regions connected to the image border,
    keeping isolated similar-colored areas (windows, shadows, etc.) intact.
    Optionally auto-detects the background color, applies chroma-key despill,
    and uses a color tolerance so slightly uneven backgrounds still key cleanly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex color of the background to remove. Used when auto_detect_color is off."
                }),
                "auto_detect_color": ("BOOLEAN", {
                    "default": True,
                    "label_on": "auto",
                    "label_off": "manual",
                    "tooltip": "If on, detect background color from image corners/borders. If off, use target_color."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.06,
                    "min": 0.0,
                    "max": 1.732,
                    "step": 0.001,
                    "tooltip": "RGB distance to target color (0..~1.732). Smaller = stricter. 0.06≈15/255"
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.08,
                    "min": 0.0,
                    "max": 1.732,
                    "step": 0.001,
                    "tooltip": "Extra RGB slack so slightly uneven background colors are still removed. Added on top of threshold. 0.08≈20/255"
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
                "despill": ("BOOLEAN", {
                    "default": True,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "Chroma-key despill: remove green/blue key-color cast from the subject fringe."
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)
    FUNCTION = "remove_black_bg_smart"
    CATEGORY = "image/alpha"

    def remove_black_bg_smart(
        self,
        image: torch.Tensor,
        target_color: str,
        threshold: float,
        feather_radius: int,
        connectivity: str,
        keep_original_alpha: bool,
        auto_detect_color: bool = True,
        despill: bool = True,
        tolerance: float = 0.08,
        onoff: bool = True,
    ):
        if not onoff:
            return (image,)

        manual_color = None if auto_detect_color else np.array(_parse_hex_color(target_color), dtype=np.float32)
        match_dist = max(float(threshold) + float(tolerance), 1e-6)

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
            target_np = _detect_bg_color(rgb_np) if manual_color is None else manual_color

            diff = rgb_np - target_np
            dist = np.sqrt(np.sum(diff * diff, axis=-1))  # [H, W]
            color_mask = dist < match_dist

            labeled, _ = ndimage.label(color_mask, structure=structure)

            border_labels = set()
            border_labels |= set(labeled[0, :].ravel())
            border_labels |= set(labeled[-1, :].ravel())
            border_labels |= set(labeled[:, 0].ravel())
            border_labels |= set(labeled[:, -1].ravel())
            border_labels.discard(0)

            remove_mask = np.isin(labeled, list(border_labels)) if border_labels else np.zeros_like(color_mask)

            expanded = ndimage.binary_dilation(remove_mask, structure=structure, iterations=2)

            color_alpha = np.clip(dist / match_dist, 0.0, 1.0).astype(np.float32)
            alpha_np = np.where(expanded, color_alpha, 1.0)

            if feather_radius > 0:
                alpha_np = ndimage.gaussian_filter(alpha_np, sigma=feather_radius)
                alpha_np = np.clip(alpha_np, 0.0, 1.0)

            if despill:
                fringe = ndimage.binary_dilation(remove_mask, structure=structure, iterations=8)
                rgb_np = _apply_despill(rgb_np, target_np, fringe)

            alpha_t = torch.from_numpy(alpha_np).to(device=image.device, dtype=image.dtype)
            rgb_t = torch.from_numpy(rgb_np).to(device=image.device, dtype=image.dtype)

            if c == 4 and keep_original_alpha:
                orig_alpha = image[i, ..., 3].clamp(0.0, 1.0)
                alpha_t = torch.min(alpha_t, orig_alpha)

            frame = torch.cat([rgb_t, alpha_t.unsqueeze(-1)], dim=-1)  # [H,W,4]
            results.append(frame)

        out = torch.stack(results, dim=0)  # [B,H,W,4]
        return (out,)

