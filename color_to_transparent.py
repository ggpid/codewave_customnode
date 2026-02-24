import torch
import numpy as np


class ColorToTransparent:
    """
    ComfyUI custom node that makes pixels matching a specific color transparent.

    Takes a hex color code and a threshold value. Pixels whose RGB distance
    to the target color is within the threshold become transparent.
    Pixels near the threshold boundary are feathered for smooth edges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "Hex color code to make transparent (e.g. #FFFFFF)"
                }),
                "threshold": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Color distance threshold (0-255). Pixels within this distance from the target color become transparent."
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Feather radius for smooth transition at the threshold boundary. 0 = hard edge."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_to_transparent"
    CATEGORY = "image/transform"
    DESCRIPTION = (
        "Makes pixels matching a specific hex color transparent. "
        "Pixels within the threshold distance from the target color become transparent. "
        "Use feather to create a smooth transition at the edge."
    )

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        if len(hex_color) != 6:
            raise ValueError(
                f"Invalid hex color: #{hex_color}. "
                f"Expected format: #RRGGBB or #RGB"
            )
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def color_to_transparent(self, image: torch.Tensor, hex_color: str, threshold: int, feather: int):
        target = np.array(self.hex_to_rgb(hex_color), dtype=np.float64)
        threshold_norm = threshold / 255.0
        feather_norm = feather / 255.0

        img_np = image.cpu().numpy().astype(np.float64)
        batch_size, height, width, channels = img_np.shape

        rgb = img_np[:, :, :, :3]

        # Euclidean distance per pixel to the target color
        diff = rgb - target[np.newaxis, np.newaxis, np.newaxis, :]
        distance = np.sqrt(np.sum(diff ** 2, axis=-1))

        if feather_norm > 0:
            # Smooth transition: fully transparent inside threshold,
            # linear ramp from transparent to opaque in the feather zone
            alpha = np.clip((distance - threshold_norm) / feather_norm, 0.0, 1.0)
        else:
            alpha = np.where(distance <= threshold_norm, 0.0, 1.0)

        # Preserve existing alpha if the input already has 4 channels
        if channels == 4:
            existing_alpha = img_np[:, :, :, 3]
            alpha = np.minimum(alpha, existing_alpha)

        output = np.zeros((batch_size, height, width, 4), dtype=np.float64)
        output[:, :, :, :3] = rgb
        output[:, :, :, 3] = alpha

        output_tensor = torch.from_numpy(output.astype(np.float32))
        output_tensor = output_tensor.to(image.device)

        return (output_tensor,)
