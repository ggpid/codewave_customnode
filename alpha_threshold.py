import torch
import numpy as np


class AlphaThreshold:
    """
    Clamp alpha values to fully opaque or fully transparent based on thresholds.

    Pixels with alpha above the upper threshold are set to 255 (fully opaque).
    Pixels with alpha below the lower threshold are set to 0 (fully transparent).
    Pixels in between are left unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upper_threshold": ("INT", {
                    "default": 240,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Alpha values above this are set to 255 (fully opaque)."
                }),
                "lower_threshold": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Alpha values below this are set to 0 (fully transparent)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)
    FUNCTION = "apply_alpha_threshold"
    CATEGORY = "image/alpha"

    def apply_alpha_threshold(self, image: torch.Tensor, upper_threshold: int, lower_threshold: int):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        b, h, w, c = image.shape
        if c != 4:
            raise ValueError(
                f"Expected RGBA image with 4 channels, got {c}. "
                "Please provide an image with an alpha channel."
            )

        upper = upper_threshold / 255.0
        lower = lower_threshold / 255.0

        out = image.clone()
        alpha = out[..., 3]

        alpha[alpha > upper] = 1.0
        alpha[alpha < lower] = 0.0

        out[..., 3] = alpha

        return (out,)
