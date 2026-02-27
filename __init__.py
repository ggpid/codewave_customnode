from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent
from .extract_mask_border import ExtractMaskBorder
from .black_bg_remove import BlackBGRemoveByDistance, BlackBGRemoveSmart

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
    "ExtractMaskBorder": ExtractMaskBorder,
    "BlackBGRemoveByDistance": BlackBGRemoveByDistance,
    "BlackBGRemoveSmart": BlackBGRemoveSmart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
    "ExtractMaskBorder": "Extract Mask Border",
    "BlackBGRemoveByDistance": "Remove Color BG (Distance)",
    "BlackBGRemoveSmart": "Remove Color BG (Smart)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
