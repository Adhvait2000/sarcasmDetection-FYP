import argparse, json
from pathlib import Path
from typing import List
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
from contextlib import nullcontext
from transformers import BlipProcessor, BlipForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def load_image_rgb(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return im.convert("RGB")

def generate_with_retry(model, processor, device, imgs: List[Image.Image], args) -> List[str]:
    """Try full batch; on OOM, retry in smaller chunks."""
    # Preprocess once for the full batch
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if (device == "cuda" and not args.no_fp16) else nullcontext())

    try:
        with torch.inference_mode():
            with amp_ctx:
                out_ids = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
        caps = [c.strip() for c in processor.batch_decode(out_ids, skip_special_tokens=True)]
        # Filter out very short captions (less than 2 words)
        caps = [c if len(c.split()) >= 2 else "" for c in caps]
        
        # Periodic memory cleanup after successful generation
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return caps
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        caps: List[str] = []
        # Improved micro-batch size: prevent very large micro-batches
        mb = max(1, min(2, len(imgs) // 2))
        print(f"[WARN] OOM on batch of {len(imgs)}; retrying with micro-batch={mb}")
        
        for s in range(0, len(imgs), mb):
            sub_imgs = imgs[s:s+mb]
            sub_inputs = processor(images=sub_imgs, return_tensors="pt").to(device)
            with torch.inference_mode():
                with amp_ctx:
                    out_ids = model.generate(
                        **sub_inputs,
                        max_length=args.max_length,
                        min_length=args.min_length,
                        num_beams=args.num_beams,
                        do_sample=False,
                    )
            sub_caps = [c.strip() for c in processor.batch_decode(out_ids, skip_special_tokens=True)]
            # Filter out very short captions for micro-batches too
            sub_caps = [c if len(c.split()) >= 2 else "" for c in sub_caps]
            caps.extend(sub_caps)
            
            # Memory cleanup after each micro-batch
            if device == "cuda":
                torch.cuda.empty_cache()
                
        return caps

def main():
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_image_root", required=True, help="Folder with images")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--model_name", default="Salesforce/blip-image-captioning-large",
                    help="BLIP model: ...-base or ...-large")
    ap.add_argument("--batch_size", type=int, default=8, help="Images per batch")
    ap.add_argument("--max_length", type=int, default=30)
    ap.add_argument("--min_length", type=int, default=5)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--no_fp16", action="store_true", help="Disable AMP (use fp32)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load BLIP
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device).eval()

    root = Path(args.raw_image_root)
    assert root.is_dir(), f"Not a directory: {root}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[WARN] No images found under {root}")
        return

    print(f"[INFO] Found {len(files)} images under {root}")
    print(f"[INFO] Model loaded: {args.model_name} (batch_size={args.batch_size}, fp16={not args.no_fp16})")

    B = max(1, args.batch_size)

    ok = fail = empty_captions = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for i in tqdm(range(0, len(files), B), desc="BLIP captioning", dynamic_ncols=True, leave=True):
            batch_paths = files[i: i + B]

            # 1) Load images
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

            # 2) Generate captions (OOM-safe)
            try:
                caps = generate_with_retry(model, processor, device, imgs, args)
            except Exception as e:
                # Unexpected error: mark the whole batch as failed
                print(f"[ERROR] Batch processing failed: {e}")
                fail += len(imgs)
                continue

            # 3) Write JSONL
            for image_id, cap in zip(img_ids, caps):
                fw.write(json.dumps({"image_id": image_id, "caption": cap}, ensure_ascii=False) + "\n")
                if cap:  # Non-empty caption
                    ok += 1
                else:
                    empty_captions += 1

    print(f"[DONE] wrote {ok} captions to {out_path}")
    print(f"[INFO] Empty/filtered captions: {empty_captions}")
    print(f"[INFO] Load failures: {fail}")

if __name__ == "__main__":
    main()