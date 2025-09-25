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

def deduplicate_similar_anps(anps: List[str], max_results: int = 15) -> List[str]:
    """
    Remove very similar ANPs to reduce redundancy while preserving diversity.
    This is a standalone version of the method from KnowledgeExtractor.
    """
    if not anps:
        return anps
        
    result = []
    seen_roots = set()
    
    # First pass: include diverse root concepts
    for anp in anps:
        # Extract the main noun (usually the last meaningful word)
        words = anp.split()
        root = words[-1] if words else anp
        
        # Skip very common words that don't help with uniqueness
        if root in {"is", "a", "an", "the", "of", "in", "at", "here", "that", "something"}:
            root = words[-2] if len(words) > 1 else root
            
        if root not in seen_roots:
            result.append(anp)
            seen_roots.add(root)
            if len(result) >= max_results:
                break
    
    # Second pass: fill remaining slots with high-quality variations
    if len(result) < max_results:
        for anp in anps:
            if anp not in result and len(result) < max_results:
                # Prefer simpler, more direct forms
                if any(simple in anp for simple in ["photo of", "image of", "a ", "the "]):
                    result.append(anp)
    
    return result

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
    sim = (image_feat.unsqueeze(0) @ anp_text_features.T).squeeze(0)  # [N]
    k = min(max_return, sim.numel())
    vals, idxs = torch.topk(sim, k=k, largest=True)
    out: List[Tuple[str, float]] = []
    for i, v in zip(idxs.tolist(), vals.tolist()):
        if v >= confidence_threshold:
            out.append((candidate_anps[i], float(v)))
    return out

def main():
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--max_per_type", type=int, default=15)  # Reduced default
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
    extractor.clip_model.eval()

    # One-time caches (on device)
    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] Building ANP cache...")
    extractor._ensure_cached_anp_text_features()
    print(f"[INFO] ANP cache complete. Building attribute cache...")
    extractor.ensure_cached_attribute_text_features()
    print(f"[INFO] All caches ready. Starting processing...")

    candidate_anps = extractor._cached_anp_strings
    anp_text_features = extractor._cached_anp_text_features  # [N, D] normalized
    attr_texts = extractor._attr_all_texts                   # list of all attr prompts
    attr_feats = extractor._attr_all_text_features           # [M, D] normalized
    anp_to_attr_indices = extractor._attr_anp_to_indices     # ANP -> list[int]

    ok = fail = 0
    B = max(1, args.batch_size)

    with out.open("w", encoding="utf-8") as fw:
        for i in tqdm(range(0, len(files), B), desc="Precomputing", dynamic_ncols=True, leave=True):
            batch_paths = files[i : i + B]

            # 1) Load images for this batch
            imgs: List[Image.Image] = []
            img_ids: List[str] = []
            for p in batch_paths:
                try:
                    im = load_image_rgb(p)
                    imgs.append(im)
                    img_ids.append(p.stem)
                except Exception:
                    fail += 1

            if not imgs:
                continue

            # 2) Encode batch images -> [B, D]
            img_inputs = extractor.clip_processor(images=imgs, return_tensors="pt").to(extractor.device)
            with torch.inference_mode():
                with extractor._autocast():
                    img_feats = extractor.clip_model.get_image_features(**img_inputs)  # [B, D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # 3) Per image: rank ANPs, then gather attribute rows and rank
            for b in range(img_feats.size(0)):
                try:
                    # ANPs - get more candidates for deduplication
                    anps_scored = rank_anps_for_features(
                        image_feat=img_feats[b],
                        candidate_anps=candidate_anps,
                        anp_text_features=anp_text_features,
                        confidence_threshold=args.confidence_threshold,
                        max_return=args.max_per_type * 3,  # Get 3x for deduplication
                    )
                    
                    # Extract ANP strings and deduplicate
                    anp_strings = [anp for anp, score in anps_scored]
                    anps_text = deduplicate_similar_anps(anp_strings, args.max_per_type)

                    # Attributes: gather rows for these ANPs
                    idxs: List[int] = []
                    for anp in anps_text:
                        idxs.extend(anp_to_attr_indices.get(anp, []))
                    if idxs:
                        sub = attr_feats[idxs]  # [K, D]
                        sim = (img_feats[b].unsqueeze(0) @ sub.T).squeeze(0)  # [K]
                        k = min(20, sim.numel())
                        vals, rel = torch.topk(sim, k=k, largest=True)
                        attrs_scored: List[Tuple[str, float]] = []
                        for sub_i, v in zip(rel.tolist(), vals.tolist()):
                            if v >= args.confidence_threshold:
                                global_idx = idxs[sub_i]
                                attrs_scored.append((attr_texts[global_idx], float(v)))
                        attrs_text = as_list_of_str(attrs_scored, args.confidence_threshold, args.max_per_type)
                    else:
                        attrs_text = []

                    rec = {"image_id": img_ids[b], "anps": anps_text, "attributes": attrs_text}
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok += 1
                except Exception:
                    fail += 1

    print(f"[DONE] wrote {ok} records to {out} (failures: {fail})")

if __name__ == "__main__":
    main()