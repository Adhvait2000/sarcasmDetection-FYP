import argparse, json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
from utils.knowledge_extractor import KnowledgeExtractor  
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

def extract_one(p: Path, extractor: KnowledgeExtractor,
                conf_thr: float, max_per_type: int) -> Dict:
    # Open image (RGB)
    with Image.open(p) as im:
        im = im.convert("RGB")

    # 1) ANPs (list of (str, score))
    anps_scored = extractor.extract_anps_with_clip(im, max_anps=max_per_type*2)
    anps_text = as_list_of_str(anps_scored, conf_thr, max_per_type)

    # 2) Attributes (list of (str, score)) — pass ANP strings
    attrs_scored = extractor.extract_attributes(im, anps_text)
    attrs_text = as_list_of_str(attrs_scored, conf_thr, max_per_type)

    return {"image_id": p.stem, "anps": anps_text, "attributes": attrs_text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--max_per_type", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="debug: process only first N")
    args = ap.parse_args()

    root = Path(args.raw_image_root)
    assert root.is_dir(), f"Not a directory: {root}"
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.limit > 0: files = files[:args.limit]
    if not files:
        print(f"[WARN] No images in {root}"); return

    extractor = KnowledgeExtractor(confidence_threshold=args.confidence_threshold)
    extractor.clip_model.eval()  # inference mode

    with out.open("w", encoding="utf-8") as fw:
        ok = fail = 0
        for i, p in enumerate(files, 1):
            try:
                rec = extract_one(p, extractor, args.confidence_threshold, args.max_per_type)
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                fail += 1
                # Optional: print(f"[ERR] {p}: {e}")
            if i % 500 == 0:
                print(f"[INFO] processed {i}/{len(files)} ...")
    print(f"[DONE] wrote {ok} records to {out} (failures: {fail})")

if __name__ == "__main__":
    main()
