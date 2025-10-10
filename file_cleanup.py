import json

def deduplicate_captions(input_file: str, output_file: str):
    """Remove duplicate captions, keeping only the version without ' 2' suffix."""
    
    seen = set()
    kept = 0
    removed = 0
    
    with open(output_file, "w", encoding="utf-8") as out:
        with open(input_file, "r", encoding="utf-8") as inp:
            for line in inp:
                if not line.strip():
                    continue
                    
                data = json.loads(line.strip())
                image_id = data["image_id"]
                
                # Remove " 2" suffix to get base ID
                base_id = image_id.replace(" 2", "")
                
                if base_id not in seen:
                    # First time seeing this base ID - keep it
                    seen.add(base_id)
                    # Always use the clean base_id (without " 2")
                    data["image_id"] = base_id
                    out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    # We've seen this base ID before - skip it
                    removed += 1
    
    print(f"[DONE] Kept: {kept}, Removed: {removed}")
    print(f"[INFO] Clean file saved as: {output_file}")

if __name__ == "__main__":
    # Simple usage
    input_file = "caches/anp_attr_all_t029.jsonl"  # Your original file
    output_file = "caches/anp_attr_all_t029_clean.jsonl"  # Clean output
    
    deduplicate_captions(input_file, output_file)