from .alpha_threshold import AlphaThreshold
from .extract_layer import ExtractTransparentLayer
from .extract_mask_border import ExtractMaskBorder
from .remove_color_bg import RemoveColorBG
from .remove_color_bg_advanced import RemoveColorBGAdvanced
from .qwen_auto_stitch import QwenAutoStitch
from .nodes_qwen import (
    TextEncodeQwenImageEditCodewave,
    TextEncodeQwenImageEditPlusCodewave,
    EmptyQwenImageLayeredLatentImageCodewave,
)

NODE_CLASS_MAPPINGS = {
    "AlphaThreshold": AlphaThreshold,
    "ExtractTransparentLayer": ExtractTransparentLayer,
    "ExtractMaskBorder": ExtractMaskBorder,
    "RemoveColorBG": RemoveColorBG,
    "RemoveColorBGAdvanced": RemoveColorBGAdvanced,
    "QwenAutoStitch": QwenAutoStitch,
    "TextEncodeQwenImageEditCodewave": TextEncodeQwenImageEditCodewave,
    "TextEncodeQwenImageEditPlusCodewave": TextEncodeQwenImageEditPlusCodewave,
    "EmptyQwenImageLayeredLatentImageCodewave": EmptyQwenImageLayeredLatentImageCodewave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaThreshold": "Alpha Threshold",
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ExtractMaskBorder": "Extract Mask Border",
    "RemoveColorBG": "Remove Color BG",
    "RemoveColorBGAdvanced": "Remove Color BG (Advanced)",
    "QwenAutoStitch": "Qwen Auto Stitch (Diff Mask)",
    "TextEncodeQwenImageEditCodewave": "Text Encode Qwen Image Edit (Codewave)",
    "TextEncodeQwenImageEditPlusCodewave": "Text Encode Qwen Image Edit Plus (Codewave)",
    "EmptyQwenImageLayeredLatentImageCodewave": "Empty Qwen Image Layered Latent (Codewave)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
