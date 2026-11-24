from .alpha_threshold import AlphaThreshold
from .extract_layer import ExtractTransparentLayer
from .extract_mask_border import ExtractMaskBorder
from .remove_color_bg import RemoveColorBG
from .qwen_auto_stitch import QwenAutoStitch
from .sam3_nodes import Sam3ModelLoaderCodewave, Sam3SegmentTiledCodewave
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
    "QwenAutoStitch": QwenAutoStitch,
    "TextEncodeQwenImageEditCodewave": TextEncodeQwenImageEditCodewave,
    "TextEncodeQwenImageEditPlusCodewave": TextEncodeQwenImageEditPlusCodewave,
    "EmptyQwenImageLayeredLatentImageCodewave": EmptyQwenImageLayeredLatentImageCodewave,
    "Sam3ModelLoaderCodewave": Sam3ModelLoaderCodewave,
    "Sam3SegmentTiledCodewave": Sam3SegmentTiledCodewave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaThreshold": "Alpha Threshold",
    "ExtractTransparentLayer": "Extract Transparent Layer",
    "ExtractMaskBorder": "Extract Mask Border",
    "RemoveColorBG": "Remove Color BG",
    "QwenAutoStitch": "Qwen Auto Stitch (Diff Mask)",
    "TextEncodeQwenImageEditCodewave": "Text Encode Qwen Image Edit (Codewave)",
    "TextEncodeQwenImageEditPlusCodewave": "Text Encode Qwen Image Edit Plus (Codewave)",
    "EmptyQwenImageLayeredLatentImageCodewave": "Empty Qwen Image Layered Latent (Codewave)",
    "Sam3ModelLoaderCodewave": "Load SAM3 Model (Codewave)",
    "Sam3SegmentTiledCodewave": "SAM3 Segment Tiled (Codewave)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
