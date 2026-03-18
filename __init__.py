from .alpha_threshold import AlphaThreshold
from .extract_layer import ExtractTransparentLayer
from .extract_mask_border import ExtractMaskBorder
from .remove_color_bg import RemoveColorBG

NODE_CLASS_MAPPINGS = {
    "AlphaThreshold": AlphaThreshold,
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ExtractMaskBorder": ExtractMaskBorder,
    "RemoveColorBG": RemoveColorBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaThreshold": "Alpha Threshold",
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ExtractMaskBorder": "Extract Mask Border",
    "RemoveColorBG": "Remove Color BG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
