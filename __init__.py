from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
