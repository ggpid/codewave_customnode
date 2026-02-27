import torch

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

