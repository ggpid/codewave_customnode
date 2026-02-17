import torch
import numpy as np


class ExtractTransparentLayer:
    """
    ComfyUI custom node that extracts the original transparent layer
    from a flattened (merged) image.

    Given:
        - A merged image (no alpha) = alpha-composited result of a
          semi-transparent single-color layer over a solid background.
        - The layer color code (hex).
        - The background color code (hex).

    The alpha compositing formula is:
        R = α * L + (1 - α) * B

    Solving for α:
        α = (R - B) / (L - B)

    Output:
        - An RGBA image with the original layer color and recovered alpha.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layer_color": ("STRING", {
                    "default": "#FF0000",
                    "multiline": False,
                    "tooltip": "Hex color code of the layer (e.g. #FF0000)"
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "Hex color code of the background (e.g. #FFFFFF)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_layer"
    CATEGORY = "image/transform"
    DESCRIPTION = (
        "Extracts the original transparent layer from an image that was created "
        "by alpha-compositing a single-color semi-transparent layer over a solid background. "
        "Recovers the per-pixel alpha values losslessly."
    )

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert a hex color string to normalized (0-1) RGB tuple."""
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

    def extract_layer(self, image: torch.Tensor, layer_color: str, background_color: str):
        """
        Extract the transparent layer from the merged image.

        Args:
            image: Input tensor of shape (B, H, W, 3) in [0, 1] range.
            layer_color: Hex color code of the layer.
            background_color: Hex color code of the background.

        Returns:
            Tuple of (output_image,) where output_image is shape (B, H, W, 4)
            with the recovered RGBA layer.
        """
        # Parse colors
        L = np.array(self.hex_to_rgb(layer_color), dtype=np.float64)
        B = np.array(self.hex_to_rgb(background_color), dtype=np.float64)

        # Convert image to numpy for processing
        img_np = image.cpu().numpy().astype(np.float64)  # (B, H, W, 3)

        batch_size, height, width, channels = img_np.shape

        # Calculate the difference between layer and background colors per channel
        diff = L - B  # shape (3,)

        # Find channels where there is a meaningful difference
        # (we can only recover alpha from channels where L != B)
        valid_channels = np.abs(diff) > 1e-10

        if not np.any(valid_channels):
            raise ValueError(
                f"Layer color ({layer_color}) and background color ({background_color}) "
                f"are identical. Cannot extract alpha information."
            )

        # Compute alpha from each valid channel:
        #   α = (R_c - B_c) / (L_c - B_c)
        # Then average across valid channels for robustness against rounding.
        # Shape of R: (B, H, W, 3)
        R = img_np  # (B, H, W, 3)

        alpha_accum = np.zeros((batch_size, height, width), dtype=np.float64)
        valid_count = 0

        for c in range(3):
            if valid_channels[c]:
                alpha_c = (R[:, :, :, c] - B[c]) / diff[c]
                alpha_accum += alpha_c
                valid_count += 1

        # Average alpha across valid channels
        alpha = alpha_accum / valid_count  # (B, H, W)

        # Clamp alpha to [0, 1]
        alpha = np.clip(alpha, 0.0, 1.0)

        # Build RGBA output: layer color + recovered alpha
        output = np.zeros((batch_size, height, width, 4), dtype=np.float64)
        output[:, :, :, 0] = L[0]  # R channel = layer red
        output[:, :, :, 1] = L[1]  # G channel = layer green
        output[:, :, :, 2] = L[2]  # B channel = layer blue
        output[:, :, :, 3] = alpha  # A channel = recovered alpha

        # Convert back to torch tensor (float32)
        output_tensor = torch.from_numpy(output.astype(np.float32))

        # Move to same device as input
        output_tensor = output_tensor.to(image.device)

        return (output_tensor,)
