from .extract_layer import ExtractTransparentLayer
from .color_to_transparent import ColorToTransparent
from .extract_windows import ExtractWindows

NODE_CLASS_MAPPINGS = {
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ColorToTransparent": ColorToTransparent,
    "ExtractWindows": ExtractWindows,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ColorToTransparent": "Color To Transparent",
    "ExtractWindows": "Extract Windows",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
