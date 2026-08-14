"""
QwenAutoStitch — ComfyUI 커스텀 노드 (단일 파일)

원본 이미지와 편집 결과를 비교해 '실제로 바뀐 영역'만 자동으로 마스크로 추출하고,
그 밖의 영역은 원본 픽셀을 그대로 되돌려 붙인다.
마스크를 직접 칠할 필요 없이 Crop & Stitch와 같은 효과를 얻는 것이 목적.

설치:
  이 패키지(codewave_node)를 ComfyUI/custom_nodes/ 에 두고 ComfyUI 재시작
  (opencv 필요: pip install opencv-python)

노드 위치: image/postprocessing → "Qwen Auto Stitch (Diff Mask)"

연결 예:
  LoadImage(원본) ──────────────┐
                                ├─→ [Qwen Auto Stitch] ─→ SaveImage
  VAEDecode(Qwen 편집 결과) ────┘                      └─→ MASK (미리보기용)
"""

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    raise ImportError("QwenAutoStitch 노드에는 opencv가 필요합니다: pip install opencv-python")


# --------------------------------------------------------------------------
# 텐서 <-> numpy 변환 (ComfyUI IMAGE = [B,H,W,C] float32 0~1, RGB)
# --------------------------------------------------------------------------
def _to_bgr(tensor_img):
    arr = (tensor_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)


# --------------------------------------------------------------------------
# 핵심 로직
# --------------------------------------------------------------------------
def align_to_reference(edited, reference, max_side=768):
    """ECC 정렬로 픽셀 드리프트(몇 px 밀림/줌) 제거."""
    ref_g = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    edt_g = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    h, w = ref_g.shape
    scale = min(1.0, max_side / max(h, w))
    ref_s = cv2.resize(ref_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edt_s = cv2.resize(edt_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ref_s = cv2.GaussianBlur(ref_s, (0, 0), 2).astype(np.float32) / 255.0
    edt_s = cv2.GaussianBlur(edt_s, (0, 0), 2).astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ref_s, edt_s, warp, cv2.MOTION_AFFINE,
                                       criteria, None, 5)
    except cv2.error:
        print("[QwenAutoStitch] ECC 정렬 실패 — 정렬 없이 진행")
        return edited

    warp[0, 2] /= scale
    warp[1, 2] /= scale
    return cv2.warpAffine(edited, warp, (reference.shape[1], reference.shape[0]),
                          flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def match_color(edited, reference, exclude_mask=None, mode="meanstd"):
    """편집본의 전역 색/노출 드리프트를 원본에 맞춰 합성 이음매를 없앤다."""
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    edt_lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mode == "median":  # 마스크가 없는 1차 패스 (이상치에 강건)
        out = edt_lab.copy()
        for c in range(3):
            out[..., c] += np.median(ref_lab[..., c] - edt_lab[..., c])
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    valid = exclude_mask < 0.1 if exclude_mask is not None else np.ones(ref_lab.shape[:2], bool)
    if valid.sum() < 1000:
        return edited

    out = edt_lab.copy()
    for c in range(3):
        r_m, r_s = ref_lab[..., c][valid].mean(), ref_lab[..., c][valid].std() + 1e-6
        e_m, e_s = edt_lab[..., c][valid].mean(), edt_lab[..., c][valid].std() + 1e-6
        out[..., c] = (edt_lab[..., c] - e_m) * (r_s / e_s) + r_m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def build_diff_mask(reference, edited, threshold, min_area_ratio, dilate, feather,
                    pre_blur=3.0):
    """LAB ΔE로 변경 영역 마스크(float 0~1) 생성."""
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    edt_lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
    if pre_blur > 0:  # 리샘플링/디테일 열화로 인한 고주파 차이 억제
        ref_lab = cv2.GaussianBlur(ref_lab, (0, 0), pre_blur)
        edt_lab = cv2.GaussianBlur(edt_lab, (0, 0), pre_blur)

    delta_e = np.sqrt(((ref_lab - edt_lab) ** 2).sum(axis=2))
    raw = (delta_e > threshold).astype(np.uint8) * 255

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_small)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_big)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    min_area = max(64, int(reference.shape[0] * reference.shape[1] * min_area_ratio))
    cleaned = np.zeros_like(raw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        cleaned = cv2.dilate(cleaned, k)

    alpha = cleaned.astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


# --------------------------------------------------------------------------
# ComfyUI 노드
# --------------------------------------------------------------------------
class QwenAutoStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 40.0, "step": 0.5}),
                "dilate": ("INT", {"default": 6, "min": 0, "max": 128}),
                "feather": ("INT", {"default": 10, "min": 0, "max": 128}),
                "min_area_ratio": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.5, "step": 0.0001}),
                "align": ("BOOLEAN", {"default": True}),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "protect_mask": ("MASK",),  # 이 영역은 무조건 원본 유지 (예: 얼굴)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "change_mask")
    FUNCTION = "stitch"
    CATEGORY = "image/postprocessing"

    def stitch(self, original, edited, threshold, dilate, feather, min_area_ratio,
               align, color_match, protect_mask=None):
        outs, masks = [], []
        batch = max(original.shape[0], edited.shape[0])

        for i in range(batch):
            ref = _to_bgr(original[min(i, original.shape[0] - 1)])
            edt = _to_bgr(edited[min(i, edited.shape[0] - 1)])

            if edt.shape[:2] != ref.shape[:2]:
                edt = cv2.resize(edt, (ref.shape[1], ref.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
            if align:
                edt = align_to_reference(edt, ref)
            if color_match:
                edt = match_color(edt, ref, mode="median")

            alpha = build_diff_mask(ref, edt, threshold, min_area_ratio, dilate, feather)

            if color_match:  # 편집 영역을 제외하고 한 번 더 정밀 매칭
                edt = match_color(edt, ref, exclude_mask=alpha, mode="meanstd")
                alpha = build_diff_mask(ref, edt, threshold, min_area_ratio, dilate, feather)

            if protect_mask is not None:  # 보호 영역은 강제로 원본
                pm = protect_mask[min(i, protect_mask.shape[0] - 1)].cpu().numpy()
                if pm.shape != alpha.shape:
                    pm = cv2.resize(pm, (alpha.shape[1], alpha.shape[0]))
                alpha = alpha * (1.0 - np.clip(pm, 0, 1))

            coverage = (alpha > 0.5).mean() * 100
            print(f"[QwenAutoStitch] 변경 영역 {coverage:.1f}%")
            if coverage > 60:
                print("[QwenAutoStitch] 경고: 전역 편집(조명/스타일)으로 보입니다. "
                      "이 노드는 국소 편집용입니다.")

            a = alpha[..., None]
            out = ref.astype(np.float32) * (1 - a) + edt.astype(np.float32) * a
            outs.append(_to_tensor(np.clip(out, 0, 255).astype(np.uint8)))
            masks.append(torch.from_numpy(alpha))

        return (torch.stack(outs), torch.stack(masks))
