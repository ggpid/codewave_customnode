from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent
from .extract_mask_border import ExtractMaskBorder
from .black_bg_remove import BlackBGRemoveByDistance

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
    "ExtractMaskBorder": ExtractMaskBorder,
    "BlackBGRemoveByDistance": BlackBGRemoveByDistance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
    "ExtractMaskBorder": "Extract Mask Border",
    "BlackBGRemoveByDistance": "Remove Black BG (Distance)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
