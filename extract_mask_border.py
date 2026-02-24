import torch
import numpy as np
from scipy import ndimage


class ExtractMaskBorder:
    """
    ComfyUI custom node that extracts the border region of a mask.

    Equivalent to the following pipeline:
        1. Invert the input mask
        2. Fill holes in the inverted mask
        3. Erode the filled mask by N iterations
        4. Compute: original_mask - invert(eroded_mask)

    The result isolates a thin strip along the boundary of transparent
    regions, useful for creating outlines or edge-based effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "침식 반복 횟수. 값이 클수록 테두리가 두꺼워집니다."
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("border_mask",)
    FUNCTION = "extract_border"
    CATEGORY = "mask"
    DESCRIPTION = (
        "Extracts the border region around transparent areas of a mask. "
        "Inverts, fills holes, erodes, then subtracts to isolate the border strip."
    )

    @staticmethod
    def _erode_float(mask: np.ndarray, iterations: int) -> np.ndarray:
        """Erode a float mask by applying min-filter repeatedly."""
        struct = ndimage.generate_binary_structure(2, 1)
        result = mask
        for _ in range(iterations):
            result = ndimage.minimum_filter(result, footprint=struct)
        return result

    def extract_border(self, mask: torch.Tensor, erode_iterations: int):
        mask_np = mask.cpu().numpy()

        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]

        results = []

        for i in range(mask_np.shape[0]):
            m = mask_np[i]

            inverted = 1.0 - m

            binary_inv = inverted > 0.5
            filled_binary = ndimage.binary_fill_holes(binary_inv)
            hole_fill_diff = filled_binary.astype(np.float32) - binary_inv.astype(np.float32)
            filled = np.clip(inverted + hole_fill_diff, 0.0, 1.0)

            eroded = self._erode_float(filled, erode_iterations)

            border = np.clip(m - (1.0 - eroded), 0.0, 1.0)
            results.append(border)

        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).to(mask.device)

        return (output_tensor,)
