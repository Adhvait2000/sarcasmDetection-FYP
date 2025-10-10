import json
from pathlib import Path

def compute_coverage(jsonl_path):
    total = 0
    non_empty = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            total += 1
            entry = json.loads(line.strip())

            # Detect type of JSONL automatically
            if "caption" in entry:
                # Caption file
                content = entry.get("caption", "").strip()
                if content and content.lower() != "none":
                    non_empty += 1

            elif "anps" in entry or "attributes" in entry:
                # ANP/Attribute file
                anps = entry.get("anps", [])
                attrs = entry.get("attributes", [])
                if (anps and any(a.strip() for a in anps)) or (attrs and any(a.strip() for a in attrs)):
                    non_empty += 1

    coverage = (non_empty / total * 100) if total > 0 else 0
    file_name = Path(jsonl_path).name
    print(f"{file_name}: {non_empty}/{total} non-empty entries ({coverage:.2f}% coverage)")
    return coverage


files = [
    "/Users/adhvaitsrinath/Documents/GitHub/sarcasmDetection-FYP/caches/anp_attr_all.jsonl",
    "/Users/adhvaitsrinath/Documents/GitHub/sarcasmDetection-FYP/caches/anp_attr_all_t027.jsonl",
    "/Users/adhvaitsrinath/Documents/GitHub/sarcasmDetection-FYP/caches/anp_attr_all_t029.jsonl",
    "/Users/adhvaitsrinath/Documents/GitHub/sarcasmDetection-FYP/caches/captions_all.jsonl",
]

coverages = {}
for path in files:
    try:
        coverages[path] = compute_coverage(path)
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
