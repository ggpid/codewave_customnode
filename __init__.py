from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent
from .extract_mask_border import ExtractMaskBorder
from .remove_color_bg import RemoveColorBG

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
    "ExtractMaskBorder": ExtractMaskBorder,
    "RemoveColorBG": RemoveColorBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
    "ExtractMaskBorder": "Extract Mask Border",
    "RemoveColorBG": "Remove Color BG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
