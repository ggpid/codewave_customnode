"""
ComfyUI nodes for SAM 3 checkpoints, including models fine-tuned with img-toolkit (jobs/<id>/ckpts/epoch_N.pt).

  * Load SAM3 Model (Codewave)          -> SAM3_MODEL
  * SAM3 Segment Tiled (Codewave)       -> image_rgba (only the segmented parts opaque), mask

Put checkpoints in ComfyUI/models/sam3/ (or give an absolute path). Needs the `sam3` package in ComfyUI's Python:
    pip install git+https://github.com/facebookresearch/sam3
"""
import os

import numpy as np
import torch
from PIL import Image

from . import sam3_core

try:
    import folder_paths

    _SAM3_DIR = os.path.join(folder_paths.models_dir, "sam3")
    _EXTS = {".pt", ".pth", ".safetensors"}
    if "sam3" in folder_paths.folder_names_and_paths:
        paths, exts = folder_paths.folder_names_and_paths["sam3"]
        if _SAM3_DIR not in paths:
            paths.append(_SAM3_DIR)
        folder_paths.folder_names_and_paths["sam3"] = (paths, set(exts) | _EXTS)
    else:
        folder_paths.folder_names_and_paths["sam3"] = ([_SAM3_DIR], _EXTS)
    os.makedirs(_SAM3_DIR, exist_ok=True)
except ImportError:  # imported outside ComfyUI (tests)
    folder_paths = None

_MODEL_CACHE = {}


def _ckpt_list():
    if folder_paths is None:
        return ["(none)"]
    names = folder_paths.get_filename_list("sam3")
    return names or ["(models/sam3 비어 있음)"]


def _device(choice):
    if choice == "auto":
        try:
            import comfy.model_management as mm
            return mm.get_torch_device()
        except ImportError:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


class Sam3ModelLoaderCodewave:
    """Load a SAM 3 checkpoint: the official sam3.pt, a SAM 3 trainer checkpoint, or an img-toolkit fine-tuning
    snapshot (jobs/<id>/ckpts/epoch_N.pt)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_ckpt_list(), {"tooltip": "ComfyUI/models/sam3/ 안의 체크포인트."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "CUDA autocast dtype. RTX 30xx 이상은 bf16 권장. CPU는 항상 fp32."}),
            },
            "optional": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "절대 경로를 넣으면 ckpt_name 대신 사용 (예: img-toolkit의 workspace/jobs/<작업>/ckpts/epoch_20.pt)."}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load"
    CATEGORY = "codewave/sam3"

    def load(self, ckpt_name, device, precision, ckpt_path=""):
        path = ckpt_path.strip().strip('"')
        if not path:
            if folder_paths is None:
                raise ValueError("ckpt_path를 지정하세요")
            path = folder_paths.get_full_path("sam3", ckpt_name)
            if path is None:
                raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: models/sam3/{ckpt_name}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
        dev = _device(device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        key = (os.path.abspath(path), os.path.getmtime(path), str(dev))
        m = _MODEL_CACHE.get(key)
        if m is None:
            _MODEL_CACHE.clear()  # one SAM 3 (~3.4 GB) at a time
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            m = sam3_core.load_model(path, dev, dtype)
            _MODEL_CACHE[key] = m
            if not m.has_masks:
                print(f"[SAM3 Codewave] {os.path.basename(path)}: segmentation head 없음 (box 전용 학습) -> 박스를 마스크로 채웁니다")
        m.dtype = dtype
        return (m,)


class Sam3SegmentTiledCodewave:
    """Segment text prompts with SAM 3 and return the image with only the segmented parts opaque.
    Large images are tiled (SAHI style) and every instance is re-segmented on a crop at original resolution, so
    small objects are not missed and edges stay sharp."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "window",
                    "tooltip": "찾을 대상 (영어 명사구). 쉼표로 여러 개: window, door"}),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "검출 신뢰도 임계값. 놓치는 게 있으면 낮추세요."}),
                "mode": (["tiled+refine", "tiled", "verified+refine", "verified", "two-stage", "single"], {
                    "default": "tiled+refine",
                    "tooltip": "tiled+refine: 타일링 + 객체별 원본 해상도 재추론 (가장 정확)\n"
                               "tiled: 전체 + 겹치는 타일 (SAHI)\n"
                               "verified(+refine): 전체 이미지(single) 결과로 '무엇이 대상인지' 판정하고, 타일 결과로 테두리만 가져옴. "
                               "single에서 안 잡힌 타일 검출은 버림 (single의 인식 + 타일의 깔끔한 테두리)\n"
                               "two-stage: 축소 이미지에서 찾고 객체별 원본 해상도 재추론\n"
                               "single: 한 번에 추론 (SAM 기본, 비교용)"}),
                "tile_size": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 64,
                    "tooltip": "타일 한 변 (px). 1024~1500 권장. 이보다 1.15배 이상 큰 이미지만 타일링합니다."}),
                "overlap": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "타일 겹침 비율. 0.2~0.3 권장."}),
            },
            "optional": {
                "merge_overlap": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05,
                    "tooltip": "두 검출이 작은 쪽 면적의 이 비율 이상 겹치면 같은 객체로 합칩니다 (타일 경계에서 잘린 객체 포함)."}),
                "refine_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "재추론 시 객체 박스 주변 여백 비율."}),
                "min_area": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 16,
                    "tooltip": "이보다 작은 객체(픽셀 수)는 버립니다."}),
                "verify_overlap": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "verified 모드: 타일 검출의 이 비율 이상이 전체 이미지 검출 안에 있어야 채택. 높일수록 엄격."}),
                "verify_margin": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "verified 모드: 전체 이미지 마스크를 이만큼(px) 넓혀서 판정·자르기. 0 = 자동 (이미지 크기 비례). "
                               "테두리가 잘리면 키우고, 옆으로 번지면 줄이세요."}),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "켜면 세그멘테이션된 부분을 투명하게 (나머지를 남김)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image_rgba", "mask", "count")
    FUNCTION = "segment"
    CATEGORY = "codewave/sam3"

    def segment(self, sam3_model, image, prompt, threshold, mode, tile_size, overlap,
                merge_overlap=0.5, refine_padding=0.15, min_area=0, invert=False, verify_overlap=0.5, verify_margin=0):
        try:
            import comfy.utils
            import comfy.model_management as mm
            interrupt = mm.throw_exception_if_processing_interrupted
        except ImportError:
            comfy, interrupt = None, None
        if image.dim() == 3:
            image = image.unsqueeze(0)
        outs, masks, count = [], [], 0
        for b in range(image.shape[0]):
            rgb = image[b, ..., :3].clamp(0, 1)
            pil = Image.fromarray((rgb.cpu().numpy() * 255.0).round().astype(np.uint8))
            pbar = comfy.utils.ProgressBar(100) if comfy is not None else None
            stats = {}

            def progress(done, total, pbar=pbar):
                if pbar is not None:
                    pbar.update_absolute(int(done * 100 / max(total, 1)), 100)
            mask, insts = sam3_core.segment(
                sam3_model, pil, prompt, threshold=threshold, mode=mode, tile_size=tile_size, overlap=overlap,
                merge_ios=merge_overlap, refine_pad=refine_padding, min_area=min_area,
                verify_frac=verify_overlap, verify_margin=verify_margin, progress=progress, interrupt=interrupt, stats=stats)
            if mode.startswith("verified") and stats.get("tiles"):
                print(f"[SAM3 Codewave] verified: {stats['tiles']} tiles, {stats['dropped']} tile detections not in the "
                      f"whole-image result dropped (margin {stats['margin']} px)")
            count += len(insts)
            m = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
            if invert:
                m = 1.0 - m
            alpha = m
            if image.shape[-1] == 4:  # keep existing transparency
                alpha = torch.minimum(alpha, image[b, ..., 3])
            outs.append(torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1))
            masks.append(m)
        return (torch.stack(outs), torch.stack(masks), count)
