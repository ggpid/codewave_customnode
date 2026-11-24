"""
SAM 3 loading + high-resolution segmentation, independent of ComfyUI (the nodes live in sam3_nodes.py).

SAM 3 stretches every input to 1008x1008 and predicts masks at a fraction of that, then upsamples them. A 4096 px
photo segmented in one pass therefore misses small objects (a window becomes a few pixels) and gets soft, blocky
edges. `segment()` combines two remedies:

  1. Tiling (SAHI style): square tiles of `tile_size` px with `overlap`, each segmented near native resolution, plus
     one pass over the whole (downscaled) image for objects larger than a tile. Detections from all passes are
     merged by mask overlap (intersection over the smaller mask), so an object cut by a tile border or found by
     several passes becomes one instance.
  2. Two-stage refinement: every merged instance is segmented again on a square crop around it at original
     resolution (when that crop is smaller than a tile, i.e. actually finer). The crop's detection that best
     overlaps the instance replaces its mask -> much sharper edges.

Checkpoints: the official `sam3.pt` (keys prefixed with `detector.`), SAM 3 trainer checkpoints ({"model": ...}),
img-toolkit fine-tuning snapshots (`ckpts/epoch_N.pt`) and plain state dicts (.pt/.pth/.safetensors).
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

SAM3_RES = 1008  # SAM 3 input resolution (position encodings are precomputed for it)


# ------------------------------------------------------------------ loading

_patched = False


def _patch_sam3():
    """SAM 3.1 uses a fused bf16 MLP kernel (sam3.perflib addmm_act) that only works on CUDA; fall back to the plain
    fc1 + activation elsewhere so CPU inference also works."""
    global _patched
    if _patched:
        return
    try:
        import torch.nn.functional as F
        import sam3.model.vitdet as vd
        orig = vd.addmm_act

        def safe_addmm_act(activation, linear, mat1):
            if torch.is_grad_enabled() or not mat1.is_cuda:
                y = linear(mat1)
                return F.relu(y) if activation in (F.relu, torch.nn.ReLU) else F.gelu(y, approximate="tanh")
            return orig(activation, linear, mat1)
        vd.addmm_act = safe_addmm_act
    except (ImportError, AttributeError):
        pass  # older sam3 without the fused kernel
    _patched = True


def _state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if any(k.startswith("detector.") for k in sd):  # official checkpoint: image detector + video tracker
        sd = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


@dataclass
class Sam3Model:
    model: object
    processor: object
    device: torch.device
    dtype: torch.dtype  # autocast dtype on CUDA (bf16 / fp16), float32 = no autocast
    has_masks: bool  # False for box-only fine-tunes (no segmentation head) -> boxes are filled as masks
    path: str
    _text: dict = field(default_factory=dict)  # prompt -> cached text features


def load_model(path, device="cuda", dtype=torch.bfloat16):
    """Build SAM 3 (image model) and load a checkpoint. Requires the `sam3` package."""
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise RuntimeError("SAM 3 패키지가 없습니다. ComfyUI 파이썬 환경에서 설치하세요: "
                           "pip install git+https://github.com/facebookresearch/sam3") from e
    _patch_sam3()
    device = torch.device(device)
    model = build_sam3_image_model(device="cpu", eval_mode=True, checkpoint_path=None, load_from_HF=False,
                                   enable_segmentation=True)
    sd = _state_dict(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("inst_interactive_predictor")]
    other = [k for k in missing if not k.startswith("segmentation_head")]
    if len(other) > 0.5 * max(1, len(model.state_dict())):
        raise RuntimeError(f"SAM 3 가중치가 아닌 것 같습니다: {path} ({len(other)} keys missing, e.g. {other[:3]})")
    has_masks = not any(k.startswith("segmentation_head") for k in missing)
    model.to(device).eval()
    proc = Sam3Processor(model, resolution=SAM3_RES, device=str(device), confidence_threshold=0.5)
    return Sam3Model(model=model, processor=proc, device=device, dtype=dtype, has_masks=has_masks, path=path)


# ------------------------------------------------------------------ one forward pass

def _autocast(m: Sam3Model):
    from contextlib import nullcontext
    if m.device.type != "cuda" or m.dtype == torch.float32:
        return nullcontext()
    return torch.autocast("cuda", dtype=m.dtype)


@dataclass
class Det:
    box: np.ndarray  # x0, y0, x1, y1 in the coordinates of the image passed to detect()
    score: float
    mask: np.ndarray  # bool HxW at that image's size (box filled when the model has no mask head)


@torch.inference_mode()
def detect(m: Sam3Model, image: Image.Image, prompt: str, threshold: float):
    """All instances of `prompt` in `image` (resized to 1008x1008 internally)."""
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate
    proc, model = m.processor, m.model
    with _autocast(m):
        state = proc.set_image(image)
        if prompt not in m._text:
            m._text[prompt] = model.backbone.forward_text([prompt], device=proc.device)
        state["backbone_out"].update(m._text[prompt])
        out = model.forward_grounding(backbone_out=state["backbone_out"], find_input=proc.find_stage,
                                      geometric_prompt=model._get_dummy_prompt(), find_target=None)
    probs = out["pred_logits"].float().sigmoid()
    if "presence_logit_dec" in out:
        probs = probs * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)
    probs = probs.squeeze(-1)
    keep = probs > threshold
    w, h = image.size
    boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][keep].float())
    boxes = (boxes * torch.tensor([w, h, w, h], device=boxes.device)).cpu().numpy()
    scores = probs[keep].cpu().numpy()
    masks = None
    if m.has_masks and out.get("pred_masks") is not None and keep.any():
        pm = out["pred_masks"][keep].float()
        masks = (interpolate(pm.unsqueeze(1), (h, w), mode="bilinear", align_corners=False).sigmoid() > 0.5)[:, 0].cpu().numpy()
    dets = []
    for i in range(len(scores)):
        if masks is not None:
            mk = masks[i]
        else:
            mk = np.zeros((h, w), bool)
            x0, y0, x1, y1 = [int(round(v)) for v in boxes[i]]
            mk[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        dets.append(Det(box=boxes[i], score=float(scores[i]), mask=mk))
    return dets


# ------------------------------------------------------------------ instances on the full canvas

@dataclass
class Inst:
    x0: int
    y0: int
    mask: np.ndarray  # bool, cropped to the mask's bounding box; covers [y0:y0+h, x0:x0+w] of the full image
    score: float
    fine: bool  # from a tile / refinement (near native resolution), False = whole-image pass

    @property
    def x1(self):
        return self.x0 + self.mask.shape[1]

    @property
    def y1(self):
        return self.y0 + self.mask.shape[0]

    @property
    def area(self):
        return int(self.mask.sum())


def _inst(mask, ox, oy, score, fine, scale=1.0):
    """Det mask (in a tile/crop/downscaled frame) -> Inst on the full canvas. `scale` maps that frame to full size."""
    if scale != 1.0:
        H, W = int(round(mask.shape[0] * scale)), int(round(mask.shape[1] * scale))
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Inst(x0=int(ox + x0), y0=int(oy + y0), mask=mask[y0:y1, x0:x1].copy(), score=score, fine=fine)


def _inter(a: Inst, b: Inst):
    x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
    if x0 >= x1 or y0 >= y1:
        return 0
    return int((a.mask[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0] & b.mask[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]).sum())


def _union(insts):
    x0, y0 = min(i.x0 for i in insts), min(i.y0 for i in insts)
    x1, y1 = max(i.x1 for i in insts), max(i.y1 for i in insts)
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for i in insts:
        m[i.y0 - y0:i.y1 - y0, i.x0 - x0:i.x1 - x0] |= i.mask
    return Inst(x0=x0, y0=y0, mask=m, score=max(i.score for i in insts), fine=any(i.fine for i in insts))


def merge(insts, ios_thr=0.5):
    """Group instances whose masks overlap by >= ios_thr of the smaller one (transitively). Each group keeps the
    union of its fine (tile) masks; the coarse whole-image mask only fills in when the fine ones cover less than half
    of it (an object part no tile detected)."""
    n = len(insts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    areas = [i.area for i in insts]
    for a in range(n):
        for b in range(a + 1, n):
            if areas[a] == 0 or areas[b] == 0:
                continue
            inter = _inter(insts[a], insts[b])
            if inter and inter / min(areas[a], areas[b]) >= ios_thr:
                parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(insts[i])
    out = []
    for g in groups.values():
        fine = [i for i in g if i.fine]
        coarse = [i for i in g if not i.fine]
        if fine and coarse:
            fu, cu = _union(fine), _union(coarse)
            out.append(fu if fu.area >= 0.5 * cu.area else _union([fu, cu]))
        else:
            out.append(_union(g))
    return out


# ------------------------------------------------------------------ tiling + refinement

def _positions(length, tile, step):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, step))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def tile_grid(W, H, tile_size, overlap):
    """Square tiles (SAM 3 stretches inputs to a square) of side min(tile_size, W, H) covering the image."""
    t = min(tile_size, W, H)
    step = max(1, int(round(t * (1.0 - overlap))))
    return [(x, y, t) for y in _positions(H, t, step) for x in _positions(W, t, step)]


def _square_crop(inst: Inst, W, H, pad_ratio, min_pad=16):
    w, h = inst.mask.shape[1], inst.mask.shape[0]
    s = int(round(max(w, h) * (1 + 2 * pad_ratio))) + 2 * min_pad
    s = min(s, W, H)
    cx, cy = inst.x0 + w / 2, inst.y0 + h / 2
    x0 = int(round(min(max(cx - s / 2, 0), W - s)))
    y0 = int(round(min(max(cy - s / 2, 0), H - s)))
    return x0, y0, s


def _iou_in_crop(inst: Inst, det_mask, x0, y0):
    """IoU of an instance and a crop-frame mask, both restricted to the crop."""
    s = det_mask.shape[0]
    sub = np.zeros((s, s), bool)
    ix0, iy0, ix1, iy1 = max(inst.x0, x0), max(inst.y0, y0), min(inst.x1, x0 + s), min(inst.y1, y0 + s)
    if ix0 < ix1 and iy0 < iy1:
        sub[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = inst.mask[iy0 - inst.y0:iy1 - inst.y0, ix0 - inst.x0:ix1 - inst.x0]
    u = (sub | det_mask).sum()
    return float((sub & det_mask).sum() / u) if u else 0.0


def segment(m: Sam3Model, image: Image.Image, prompts, threshold=0.5, mode="tiled+refine", tile_size=1024,
            overlap=0.25, merge_ios=0.5, refine_pad=0.15, refine_match=0.3, min_area=0, max_refine=500,
            progress=None, interrupt=None):
    """Union mask (bool HxW, full resolution) of every instance of `prompts`, plus the instance list.

    mode: "single"      one pass on the whole image (SAM 3 default behaviour, for comparison)
          "tiled"       whole-image pass + overlapping tiles, merged
          "two-stage"   whole-image pass, then each instance re-segmented on a crop at original resolution
          "tiled+refine" tiles, then each instance re-segmented on a crop (most accurate, slowest)
    progress(done, total) and interrupt() are optional callbacks (ComfyUI progress bar / cancel).
    """
    image = image.convert("RGB")
    W, H = image.size
    prompts = [p.strip() for p in (prompts.split(",") if isinstance(prompts, str) else prompts) if p.strip()]
    if not prompts:
        raise ValueError("프롬프트를 입력하세요 (쉼표로 여러 개)")
    use_tiles = mode in ("tiled", "tiled+refine") and max(W, H) > tile_size * 1.15
    use_refine = mode in ("two-stage", "tiled+refine")
    tiles = tile_grid(W, H, tile_size, overlap) if use_tiles else []

    # the whole-image pass runs on a downscaled copy: SAM resizes to 1008 anyway, this just saves transfer/memory
    g_scale = max(W, H) / min(max(W, H), 2048)
    g_img = image if g_scale == 1.0 else image.resize((round(W / g_scale), round(H / g_scale)), Image.BILINEAR)

    total = len(prompts) * (1 + len(tiles))
    done = [0]

    def step():
        done[0] += 1
        if progress:
            progress(done[0], total)
        if interrupt:
            interrupt()

    all_insts = []
    for p in prompts:
        insts = []
        for d in detect(m, g_img, p, threshold):
            i = _inst(d.mask, 0, 0, d.score, fine=False, scale=g_scale)
            if i is not None:
                insts.append(i)
        step()
        for (x, y, t) in tiles:
            crop = image.crop((x, y, x + t, y + t))
            for d in detect(m, crop, p, threshold):
                i = _inst(d.mask, x, y, d.score, fine=True)
                if i is not None:
                    insts.append(i)
            step()
        insts = merge(insts, merge_ios) if insts else []

        if use_refine and insts:
            todo = [k for k, i in enumerate(insts) if _square_crop(i, W, H, refine_pad)[2] < tile_size][:max_refine]
            total += len(todo)
            for k in todo:
                inst = insts[k]
                x0, y0, s = _square_crop(inst, W, H, refine_pad)
                dets = detect(m, image.crop((x0, y0, x0 + s, y0 + s)), p, threshold)
                best, best_iou = None, refine_match
                for d in dets:
                    iou = _iou_in_crop(inst, d.mask, x0, y0)
                    if iou > best_iou:
                        best, best_iou = d, iou
                if best is not None:
                    # keep any part of the old mask outside the crop (crop clamped at the image border)
                    outside = inst.mask.copy()
                    cx0, cy0 = max(0, x0 - inst.x0), max(0, y0 - inst.y0)
                    outside[cy0:max(cy0, y0 + s - inst.y0), cx0:max(cx0, x0 + s - inst.x0)] = False
                    new = _inst(best.mask, x0, y0, max(best.score, inst.score), fine=True)
                    parts = [new] + ([Inst(inst.x0, inst.y0, outside, inst.score, True)] if outside.any() else [])
                    if new is not None:
                        insts[k] = _union(parts) if len(parts) > 1 else new
                step()
        all_insts += [i for i in insts if i.area >= min_area]

    full = np.zeros((H, W), bool)
    for i in all_insts:
        full[i.y0:i.y1, i.x0:i.x1] |= i.mask
    return full, all_insts
