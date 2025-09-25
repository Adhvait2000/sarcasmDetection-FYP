import argparse, json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image, ImageFile
from tqdm import tqdm
import torch

from utils.knowledge_extractor import KnowledgeExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def as_list_of_str(
    items: Optional[Union[List[str], List[Tuple[str, float]]]],
    confidence_threshold: float,
    max_per_type: int,
) -> List[str]:
    if not items:
        return []
    cleaned = []
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 1:
            token = str(it[0]).strip()
            score = float(it[1]) if len(it) > 1 else 1.0
            if token and score >= confidence_threshold:
                cleaned.append(token)
        else:
            token = str(it).strip()
            if token:
                cleaned.append(token)
    seen, uniq = set(), []
    for t in cleaned:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[:max_per_type]

def load_image_rgb(p: Path):
    with Image.open(p) as im:
        return im.convert("RGB")

def rank_anps_for_features(
    image_feat: torch.Tensor,
    candidate_anps: List[str],
    anp_text_features: torch.Tensor,
    confidence_threshold: float,
    max_return: int,
) -> List[Tuple[str, float]]:
    """
    image_feat: [D] normalized
    anp_text_features: [N, D] normalized
    """
    sim = (image_feat.unsqueeze(0) @ anp_text_features.T).squeeze(0)  # [N]
    k = min(max_return, sim.numel())
    topk = torch.topk(sim, k=k, largest=True)
    out: List[Tuple[str, float]] = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        if score >= confidence_threshold:
            out.append((candidate_anps[idx], float(score)))
    return out

def build_attribute_candidates(emotional: List[str], stylistic: List[str], anps: List[str]) -> List[str]:
    all_attrs = emotional + stylistic
    cands: List[str] = []
    for anp in anps:
        for attr in all_attrs:
            cands.append(f"{attr} {anp}")
    return cands

def rank_attributes_for_features(
    image_feat: torch.Tensor,
    attribute_candidates: List[str],
    encode_texts_to_device_tensor,  # extractor._encode_texts_to_device_tensor
    confidence_threshold: float,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    image_feat: [D] normalized
    """
    if not attribute_candidates:
        return []
    text_feats = encode_texts_to_device_tensor(attribute_candidates)  # [M, D], normalized
    sim = (image_feat.unsqueeze(0) @ text_feats.T).squeeze(0)  # [M]
    k = min(top_k, sim.numel())
    topk = torch.topk(sim, k=k, largest=True)
    out: List[Tuple[str, float]] = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        if score >= confidence_threshold:
            out.append((attribute_candidates[idx], float(score)))
    return out

def main():
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--max_per_type", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="debug: process only first N")
    ap.add_argument("--batch_size", type=int, default=32, help="image batch size for CLIP")
    args = ap.parse_args()

    root = Path(args.raw_image_root)
    assert root.is_dir(), f"Not a directory: {root}"
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[WARN] No images in {root}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    extractor = KnowledgeExtractor(
        confidence_threshold=args.confidence_threshold,
        device=device
    )
    extractor.clip_model.eval()  # paranoia

    # Pre-cache ANP text features on device
    extractor._ensure_cached_anp_text_features()
    candidate_anps = extractor._cached_anp_strings
    anp_text_features = extractor._cached_anp_text_features  # [N, D] on device

    # Shorthands for attributes
    emotional = extractor.emotional_attributes
    stylistic = extractor.stylistic_attributes

    ok = fail = 0
    batch_size = max(1, args.batch_size)

    with out.open("w", encoding="utf-8") as fw:
        # Iterate in batches of file paths
        for i in tqdm(range(0, len(files), batch_size), desc="Precomputing"):
            batch_paths = files[i : i + batch_size]

            # 1) Load images (skip failures)
            imgs: List[Image.Image] = []
            img_ids: List[str] = []
            valid_idx: List[int] = []
            for j, p in enumerate(batch_paths):
                try:
                    im = load_image_rgb(p)
                    imgs.append(im)
                    img_ids.append(p.stem)
                    valid_idx.append(j)
                except Exception:
                    fail += 1

            if not imgs:
                continue  # all failed

            # 2) Encode images in one go -> [B, D]
            img_inputs = extractor.clip_processor(images=imgs, return_tensors="pt").to(extractor.device)
            with torch.inference_mode():
                with extractor._autocast():
                    img_feats = extractor.clip_model.get_image_features(**img_inputs)  # [B, D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # 3) For each image in batch: rank ANPs, then attributes, write JSONL
            for b in range(img_feats.size(0)):
                try:
                    # ANPs
                    anps_scored = rank_anps_for_features(
                        image_feat=img_feats[b],
                        candidate_anps=candidate_anps,
                        anp_text_features=anp_text_features,
                        confidence_threshold=args.confidence_threshold,
                        max_return=args.max_per_type * 2,  # extra then filter
                    )
                    anps_text = as_list_of_str(anps_scored, args.confidence_threshold, args.max_per_type)

                    # Attributes (candidates depend on this image's ANPs)
                    attr_cands = build_attribute_candidates(emotional, stylistic, anps_text)
                    attrs_scored = rank_attributes_for_features(
                        image_feat=img_feats[b],
                        attribute_candidates=attr_cands,
                        encode_texts_to_device_tensor=extractor._encode_texts_to_device_tensor,
                        confidence_threshold=args.confidence_threshold,
                        top_k=20,
                    )
                    attrs_text = as_list_of_str(attrs_scored, args.confidence_threshold, args.max_per_type)

                    rec = {"image_id": img_ids[b], "anps": anps_text, "attributes": attrs_text}
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok += 1
                except Exception:
                    fail += 1

    print(f"[DONE] wrote {ok} records to {out} (failures: {fail})")

if __name__ == "__main__":
    main()
