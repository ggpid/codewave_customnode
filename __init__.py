from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent
from .extract_mask_border import ExtractMaskBorder

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
    "ExtractMaskBorder": ExtractMaskBorder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
    "ExtractMaskBorder": "Extract Mask Border",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
