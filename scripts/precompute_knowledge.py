import argparse, json
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
from utils.knowledge_extractor import KnowledgeExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------- helpers ----------
def deduplicate_similar_anps(anps: List[str], max_results: int = 15) -> List[str]:
    if not anps:
        return anps
    result, seen_roots = [], set()
    for anp in anps:
        words = anp.split()
        root = words[-1] if words else anp
        if root in {"is", "a", "an", "the", "of", "in", "at", "here", "that", "something"}:
            root = words[-2] if len(words) > 1 else root
        if root not in seen_roots:
            result.append(anp)
            seen_roots.add(root)
            if len(result) >= max_results:
                break
    if len(result) < max_results:
        for anp in anps:
            if anp not in result and len(result) < max_results:
                if any(simple in anp for simple in ["photo of", "image of", "a ", "the "]):
                    result.append(anp)
    return result


def load_image_rgb(p: Path):
    with Image.open(p) as im:
        return im.convert("RGB")


def rank_topk_with_scores(
    image_feat: torch.Tensor,
    labels: List[str],
    text_features: torch.Tensor,
    k: int,
) -> List[Tuple[str, float]]:
    sim = (image_feat.unsqueeze(0) @ text_features.T).squeeze(0)
    k = min(k, sim.numel())
    vals, idxs = torch.topk(sim, k=k, largest=True)
    return [(labels[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


def apply_threshold_limit(pairs: List[Tuple[str, float]], thr: float, max_n: int) -> List[str]:
    seen, out = set(), []
    for lab, s in pairs:
        if s >= thr and lab not in seen:
            seen.add(lab)
            out.append(lab)
            if len(out) >= max_n:
                break
    return out


def collect_done_ids(paths: List[Path]) -> set:
    done = set()
    for p in paths:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)["image_id"])
                        except Exception:
                            pass
            except Exception:
                pass
    return done


# ---------- main ----------
def main():
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", required=True,
                    help="Emit multiple thresholds in one pass, e.g. --thresholds 0.25 0.30 0.40")
    ap.add_argument("--max_per_type", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(args.raw_image_root)
    assert root.is_dir(), f"Not a directory: {root}"

    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[WARN] No images in {root}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Found {len(files)} image files")

    extractor = KnowledgeExtractor(confidence_threshold=0.0, device=device)
    extractor.clip_model.eval()
    print("[INFO] Building caches...")
    extractor._ensure_cached_anp_text_features()
    extractor.ensure_cached_attribute_text_features()
    print("[INFO] Caches ready.")

    candidate_anps = extractor._cached_anp_strings
    anp_text_features = extractor._cached_anp_text_features
    attr_texts = extractor._attr_all_texts
    attr_feats = extractor._attr_all_text_features
    anp_to_attr_indices = extractor._attr_anp_to_indices

    base_out = Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)
    thrs = sorted(set(args.thresholds))

    writers, out_paths = {}, {}
    for t in thrs:
        p = base_out.with_name(f"{base_out.stem}_t{int(round(t*100)):03d}{base_out.suffix}")
        out_paths[t] = p
        writers[t] = p.open("a", encoding="utf-8")

    done_ids = collect_done_ids(list(out_paths.values())) if args.resume else set()
    if args.resume:
        print(f"[INFO] Resume enabled. Found {len(done_ids)} already processed.")

    ok = fail = 0
    B = max(1, args.batch_size)

    for i in tqdm(range(0, len(files), B), desc="Precomputing", dynamic_ncols=True, leave=True):
        batch_paths = files[i : i + B]
        if args.resume:
            batch_paths = [p for p in batch_paths if p.stem not in done_ids]
            if not batch_paths:
                continue

        imgs, img_ids = [], []
        seen_ids = set() 
        for p in batch_paths:
            base_id = p.stem.split()[0]  # get part before any space
            if base_id in seen_ids:
                continue  # skip duplicates like "12345 2"
            seen_ids.add(base_id)

            try:
                im = load_image_rgb(p)
                imgs.append(im)
                img_ids.append(base_id)
            except Exception:
                fail += 1

        if not imgs:
            continue

        img_inputs = extractor.clip_processor(images=imgs, return_tensors="pt").to(extractor.device)
        with torch.inference_mode():
            with extractor._autocast():
                img_feats = extractor.clip_model.get_image_features(**img_inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        for b in range(img_feats.size(0)):
            try:
                anps_scored = rank_topk_with_scores(
                    image_feat=img_feats[b],
                    labels=candidate_anps,
                    text_features=anp_text_features,
                    k=args.max_per_type * 3,
                )
                anp_strings = [lab for lab, _ in anps_scored]
                anps_diverse = deduplicate_similar_anps(anp_strings, args.max_per_type)

                idxs = []
                for anp in anps_diverse:
                    idxs.extend(anp_to_attr_indices.get(anp, []))

                attrs_scored = []
                if idxs:
                    sub = attr_feats[idxs]
                    sim = (img_feats[b].unsqueeze(0) @ sub.T).squeeze(0)
                    vals, rel = torch.topk(sim, k=min(50, sim.numel()), largest=True)
                    attrs_scored = [(attr_texts[idxs[i]], float(v)) for i, v in zip(rel.tolist(), vals.tolist())]

                for t, fh in writers.items():
                    anps_t = apply_threshold_limit(anps_scored, t, args.max_per_type)
                    attrs_t = apply_threshold_limit(attrs_scored, t, args.max_per_type)
                    rec = {"image_id": img_ids[b], "anps": anps_t, "attributes": attrs_t}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                ok += 1
            except Exception:
                fail += 1

    for fh in writers.values():
        fh.close()

    print(f"[DONE] wrote {ok} records (failures: {fail}) to:")
    for p in out_paths.values():
        print(" -", p)


if __name__ == "__main__":
    main()
