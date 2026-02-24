import torch
import numpy as np
from scipy import ndimage


class ExtractWindows:
    """
    ComfyUI custom node that extracts window regions from a building image.

    Takes a building photo and a structure mask where opaque pixels represent
    the building surface (excluding windows and background). Identifies windows
    as enclosed transparent holes within the building boundary using flood-fill,
    then outputs the original image with only window areas opaque.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "building_image": ("IMAGE",),
                "structure_mask": ("IMAGE",),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold for binarizing the structure mask. "
                               "Pixels above this value are considered building structure."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_windows"
    CATEGORY = "image/transform"
    DESCRIPTION = (
        "Extracts window regions from a building image using a structure mask. "
        "The mask's opaque areas represent the building surface (walls, roof, etc.), "
        "and this node identifies windows as the enclosed transparent holes inside "
        "the building boundary, outputting only those regions from the original image."
    )

    def extract_windows(
        self,
        building_image: torch.Tensor,
        structure_mask: torch.Tensor,
        mask_threshold: float,
    ):
        img_np = building_image.cpu().numpy().astype(np.float64)
        mask_np = structure_mask.cpu().numpy().astype(np.float64)

        batch_size, height, width, img_channels = img_np.shape
        _, _, _, mask_channels = mask_np.shape

        if mask_channels >= 4:
            structure = mask_np[:, :, :, 3]
        else:
            structure = np.mean(mask_np[:, :, :, :3], axis=-1)

        structure_binary = structure > mask_threshold

        windows_mask = np.zeros((batch_size, height, width), dtype=np.float64)

        for b in range(batch_size):
            filled = ndimage.binary_fill_holes(structure_binary[b])
            windows = filled & ~structure_binary[b]
            windows_mask[b] = windows.astype(np.float64)

        output = np.zeros((batch_size, height, width, 4), dtype=np.float64)
        output[:, :, :, :3] = img_np[:, :, :, :3]
        output[:, :, :, 3] = windows_mask

        output_tensor = torch.from_numpy(output.astype(np.float32))
        output_tensor = output_tensor.to(building_image.device)

        return (output_tensor,)
