import math

import torch
import node_helpers
import comfy.utils
import comfy.model_management
import nodes


class TextEncodeQwenImageEditCodewave:
    """
    Codewave variant of TextEncodeQwenImageEdit.
    Uses the original reference image size instead of forcing ~1024x1024 area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "codewave/conditioning"

    def encode(self, clip, prompt, vae=None, image=None):
        ref_latent = None
        if image is None:
            images = []
        else:
            # Keep original image size (no 1024*1024 area resize).
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [ref_latent]}, append=True
            )
        return (conditioning,)


class TextEncodeQwenImageEditPlusCodewave:
    """
    Codewave variant of TextEncodeQwenImageEditPlus.
    VL path still uses ~384x384; VAE reference latents keep original image size
    instead of forcing ~1024x1024 area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "codewave/conditioning"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency "
            "with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    # Keep original image size for reference latents (no 1024*1024 area resize).
                    ref_latents.append(vae.encode(image[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return (conditioning,)


class EmptyQwenImageLayeredLatentImageCodewave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 640, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "layers": ("INT", {"default": 3, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "codewave/latent"

    def generate(self, width, height, layers, batch_size=1):
        latent = torch.zeros(
            [batch_size, 16, layers + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)
