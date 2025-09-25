import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Optional, Dict, Set
from contextlib import nullcontext
import numpy as np
from PIL import Image

class KnowledgeExtractor:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = 0.7,
        device: str = "cpu",
    ):
        """
        Knowledge extractor with:
          - GPU/AMP support
          - Cached ANP text features (one-time)
          - On-device text feature cache for arbitrary prompts
          - One-time global attribute prompt features & ANP->indices map
          - Optimized deduplication and quality filtering
        """
        self.device = torch.device(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.confidence_threshold = confidence_threshold

        # Prioritized ANP templates - simpler, more natural ones first
        self.anp_templates = [
            # Tier 1: Most natural/direct
            "{}",
            "a {}",
            "an {}",
            "the {}",
            
            # Tier 2: Photo context
            "photo of {}",
            "image of {}",
            "a photo of {}",
            "an image of {}",
            "a picture of {}",
            
            # Tier 3: Spatial context
            "{} in the image",
            "{} in the photo",
            "{} in the scene",
            "this {}",
            "this is a {}",
            "a {} here",
            
            # Tier 4: More complex
            "showing a {}",
            "contains a {}",
            "looking at a {}",
            
            # Tier 5: Category-specific (filtered)
            "a {} scene",
            "a {} object",
            "a {} doing something",
            "a {} that is"
        ]

        # Define problematic template-noun combinations to skip
        self.skip_combinations = {
            # Cross-category combinations that don't make sense
            ("person", "food"), ("person", "building"), ("person", "vehicle"),
            ("food", "person"), ("food", "building"), ("food", "vehicle"), ("food", "animal"),
            ("building", "person"), ("building", "food"), ("building", "animal"),
            ("vehicle", "person"), ("vehicle", "food"), ("vehicle", "animal"),
            ("animal", "food"), ("animal", "building"), ("animal", "vehicle"),
            ("object", "person"), ("scene", "person")
        }

        # Enhanced attributes
        self.emotional_attributes = [
            # Basic emotions
            "happy", "sad", "angry", "surprised", "confused", "excited", 
            "bored", "worried", "proud", "embarrassed", "amused", "frustrated",
            
            # Extended emotions
            "joyful", "cheerful", "melancholy", "depressed", "furious", "calm",
            "peaceful", "anxious", "nervous", "confident", "shy", "bold",
            "serious", "playful", "mysterious", "friendly", "hostile"
        ]
        
        self.stylistic_attributes = [
            # Visual properties
            "bright", "dark", "colorful", "monochrome", "black and white",
            "blurry", "sharp", "clear", "focused", "out of focus",
            
            # Style/Era
            "modern", "vintage", "retro", "contemporary", "classic", "old",
            "new", "ancient", "futuristic", "traditional",
            
            # Condition/State
            "clean", "dirty", "messy", "organized", "neat", "broken", "damaged",
            "worn", "shiny", "dull", "smooth", "rough", "wet", "dry",
            
            # Formality
            "casual", "formal", "elegant", "simple", "complex", "fancy", "plain",
            
            # Size/Scale
            "big", "small", "large", "tiny", "huge", "massive", "mini",
            
            # Quality/Aesthetic
            "beautiful", "ugly", "pretty", "gorgeous", "stunning", "attractive",
            "artistic", "creative", "professional", "amateur"
        ]
        
        # Physical/descriptive attributes
        self.physical_attributes = [
            "tall", "short", "wide", "narrow", "thick", "thin", "heavy", "light",
            "round", "square", "rectangular", "circular", "curved", "straight",
            "soft", "hard", "flexible", "rigid", "transparent", "opaque",
            "metallic", "wooden", "plastic", "glass", "fabric", "leather"
        ]
        
        # ===== Text caches =====
        self._cached_anp_strings: Optional[List[str]] = None
        self._cached_anp_text_features: Optional[torch.Tensor] = None  # [N, D] on device

        # Global attribute prompts & features
        self._attr_all_texts: Optional[List[str]] = None
        self._attr_all_text_features: Optional[torch.Tensor] = None  # [M, D] on device
        self._attr_anp_to_indices: Optional[Dict[str, List[int]]] = None

        # On-device cache for arbitrary text → feature row (normalized)
        self._text_feature_cache: Dict[str, torch.Tensor] = {}

    # ---------- Utilities ----------

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda" else nullcontext()
        )

    def _should_skip_combination(self, template: str, noun: str) -> bool:
        """Check if template-noun combination should be skipped."""
        # Extract category keywords from template
        template_categories = []
        if "person" in template:
            template_categories.append("person")
        if "food" in template:
            template_categories.append("food")
        if "building" in template:
            template_categories.append("building")
        if "vehicle" in template:
            template_categories.append("vehicle")
        if "animal" in template:
            template_categories.append("animal")
        if "object" in template:
            template_categories.append("object")
        if "scene" in template:
            template_categories.append("scene")
        
        # Check noun category
        noun_category = self._get_noun_category(noun)
        
        # Skip if any template category conflicts with noun category
        for template_cat in template_categories:
            if (template_cat, noun_category) in self.skip_combinations:
                return True
        return False

    def _get_noun_category(self, noun: str) -> str:
        """Categorize nouns to help filter combinations."""
        people_nouns = {"person", "man", "woman", "child", "baby", "boy", "girl", "adult", 
                       "teenager", "individual", "human"}
        food_nouns = {"food", "pizza", "burger", "sandwich", "cake", "bread", "fruit", 
                     "apple", "banana", "orange", "vegetable", "salad", "meat", "chicken", 
                     "fish", "drink", "coffee", "tea", "water", "wine", "beer", "meal"}
        building_nouns = {"house", "building", "office", "school", "hospital", "church", 
                         "store", "restaurant", "hotel", "bridge", "tower", "castle"}
        vehicle_nouns = {"car", "truck", "bike", "bicycle", "motorcycle", "bus", "train", 
                        "plane", "airplane", "helicopter", "boat", "ship", "vehicle", "scooter", "van"}
        animal_nouns = {"dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", 
                       "chicken", "elephant", "lion", "tiger", "bear", "monkey", "rabbit", 
                       "deer", "wolf", "animal", "pet", "wildlife", "insect", "butterfly", "bee"}
        
        if noun in people_nouns:
            return "person"
        elif noun in food_nouns:
            return "food"
        elif noun in building_nouns:
            return "building"
        elif noun in vehicle_nouns:
            return "vehicle"
        elif noun in animal_nouns:
            return "animal"
        else:
            return "object"

    def _deduplicate_similar_anps(self, anps: List[str], max_results: int = 15) -> List[str]:
        """Remove very similar ANPs to reduce redundancy while preserving diversity."""
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

    def _generate_anp_candidates(self) -> List[str]:
        """Generate candidate ANPs using templates and common nouns with filtering."""
        candidates = []
        common_nouns = [
            # People
            "person", "man", "woman", "child", "baby", "boy", "girl", "adult", "teenager",
            "crowd", "group", "family", "couple", "individual", "human", "people",
            
            # Animals
            "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "chicken",
            "elephant", "lion", "tiger", "bear", "monkey", "rabbit", "deer", "wolf",
            "animal", "pet", "wildlife", "insect", "butterfly", "bee",
            
            # Vehicles
            "car", "truck", "bike", "bicycle", "motorcycle", "bus", "train", "plane",
            "airplane", "helicopter", "boat", "ship", "vehicle", "scooter", "van",
            
            # Buildings/Places
            "house", "building", "office", "school", "hospital", "church", "store",
            "restaurant", "hotel", "bridge", "tower", "castle", "barn", "garage",
            "room", "kitchen", "bedroom", "bathroom", "street", "road", "park",
            
            # Nature
            "tree", "flower", "grass", "mountain", "river", "ocean", "lake", "forest",
            "beach", "sky", "cloud", "sun", "moon", "rock", "stone", "hill", "field",
            "garden", "plant", "leaf", "branch", "water", "fire", "smoke",
            
            # Food
            "food", "pizza", "burger", "sandwich", "cake", "bread", "fruit", "apple",
            "banana", "orange", "vegetable", "salad", "meat", "chicken", "fish",
            "drink", "coffee", "tea", "water", "wine", "beer", "meal",
            
            # Objects
            "phone", "computer", "laptop", "book", "chair", "table", "bed", "sofa",
            "television", "tv", "screen", "camera", "clock", "lamp", "bottle", "cup",
            "glass", "plate", "knife", "fork", "bag", "box", "ball", "toy", "game",
            "tool", "machine", "device", "equipment", "instrument",
            
            # Clothing/Accessories
            "shirt", "pants", "dress", "shoes", "hat", "jacket", "coat", "glasses",
            "watch", "jewelry", "ring", "necklace", "purse", "clothes",
            
            # Abstract/General
            "object", "thing", "item", "stuff", "material", "surface", "shape",
            "color", "pattern", "texture", "design", "art", "artwork", "painting",
            "sign", "text", "number", "letter", "symbol", "logo", "brand"
        ]
        
        for template in self.anp_templates:
            for noun in common_nouns:
                # Skip problematic combinations
                if not self._should_skip_combination(template, noun):
                    candidates.append(template.format(noun))
        
        return candidates

    def _encode_texts_to_device_tensor(self, texts: List[str], chunk: int = 512) -> torch.Tensor:
        """
        Encode texts to normalized features on self.device with caching.
        Returns [N, D].
        """
        new_texts = [t for t in texts if t not in self._text_feature_cache]
        if new_texts:
            for i in range(0, len(new_texts), chunk):
                batch = new_texts[i:i+chunk]
                inputs = self.clip_processor(text=batch, return_tensors="pt", padding=True).to(self.device)
                with torch.inference_mode():
                    with self._autocast():
                        feats = self.clip_model.get_text_features(**inputs)  # [B, D]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                for t, row in zip(batch, feats):
                    # detach but keep on device
                    self._text_feature_cache[t] = row.detach()
        rows = [self._text_feature_cache[t].unsqueeze(0) for t in texts]
        return torch.cat(rows, dim=0) if rows else torch.empty(0, device=self.device)

    # ---------- One-time caches ----------

    def _ensure_cached_anp_text_features(self):
        """Encode ANP candidates once and cache their text features on device."""
        if self._cached_anp_text_features is not None:
            return
        self._cached_anp_strings = self._generate_anp_candidates()
        text_feats = self._encode_texts_to_device_tensor(self._cached_anp_strings)  # [N, D]
        self._cached_anp_text_features = text_feats  # normalized on device

    def ensure_cached_attribute_text_features(self):
        """
        Build global attribute prompts (attrs × all ANP candidates) once,
        encode to features on device, and build ANP→indices mapping.
        """
        if self._attr_all_text_features is not None:
            return
        self._ensure_cached_anp_text_features()
        attrs = self.emotional_attributes + self.stylistic_attributes + self.physical_attributes

        all_texts: List[str] = []
        anp_to_indices: Dict[str, List[int]] = {}
        for anp in self._cached_anp_strings:
            start = len(all_texts)
            for a in attrs:
                all_texts.append(f"{a} {anp}")
            end = len(all_texts)
            anp_to_indices[anp] = list(range(start, end))

        feats = self._encode_texts_to_device_tensor(all_texts)  # [M, D], normalized, on device
        self._attr_all_texts = all_texts
        self._attr_all_text_features = feats
        self._attr_anp_to_indices = anp_to_indices

    # ---------- Image helpers ----------

    def _tensor_to_pil(self, tensor) -> Optional[Image.Image]:
        """
        Convert torch Tensor to PIL Image.
        Accepts [C,H,W] or [H,W,C], float [0..1] or [0..255], or uint8.
        """
        if not torch.is_tensor(tensor):
            return None
        t = tensor.detach().cpu()
        if t.ndim == 3:
            if t.shape[0] in (1, 3):  # [C,H,W] → [H,W,C]
                t = t.permute(1, 2, 0)
        else:
            return None
        if t.dtype in (torch.float16, torch.float32, torch.float64):
            arr = t.numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        elif t.dtype == torch.uint8:
            arr = t.numpy()
        else:
            arr = t.to(torch.uint8).numpy()
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return Image.fromarray(arr, mode="RGB")

    # ---------- Public APIs ----------

    def extract_anps_with_clip(self, image, max_anps=15) -> List[Tuple[str, float]]:
        """Rank ANP candidates against image and return top with scores."""
        if torch.is_tensor(image):
            image = self._tensor_to_pil(image)
            if image is None:
                return []

        self._ensure_cached_anp_text_features()
        candidate_anps = self._cached_anp_strings
        text_features = self._cached_anp_text_features  # [N, D] normalized

        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            with self._autocast():
                img_feat = self.clip_model.get_image_features(**img_inputs)  # [1, D]
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_features.T).squeeze(0)  # [N]

        # Get more candidates initially for deduplication
        k = min(max_anps * 3, sim.numel())
        vals, idxs = torch.topk(sim, k=k, largest=True)
        
        # Filter by confidence and collect candidates
        candidates_with_scores = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            if v >= self.confidence_threshold:
                candidates_with_scores.append((candidate_anps[i], float(v)))
        
        # Extract just the ANP strings for deduplication
        candidate_anps_only = [anp for anp, score in candidates_with_scores]
        
        # Deduplicate while preserving order and diversity
        deduplicated_anps = self._deduplicate_similar_anps(candidate_anps_only, max_anps)
        
        # Reconstruct final results with scores
        anp_to_score = {anp: score for anp, score in candidates_with_scores}
        result = []
        for anp in deduplicated_anps:
            if anp in anp_to_score:
                result.append((anp, anp_to_score[anp]))
        
        return result

    def extract_attributes(self, image, anps: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Use precomputed global attribute features.
        For selected ANPs, gather their attribute rows and score against image.
        """
        if not anps:
            return []
        if torch.is_tensor(image):
            image = self._tensor_to_pil(image)
            if image is None:
                return []

        self.ensure_cached_attribute_text_features()
        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            with self._autocast():
                img_feat = self.clip_model.get_image_features(**img_inputs)  # [1, D]
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [1, D]

        # Gather attribute rows for these ANPs
        idxs: List[int] = []
        for anp in anps:
            idxs.extend(self._attr_anp_to_indices.get(anp, []))
        if not idxs:
            return []

        sub_feats = self._attr_all_text_features[idxs]          # [K, D]
        sim = (img_feat @ sub_feats.T).squeeze(0)               # [K]

        k = min(top_k, sim.numel())
        vals, rel = torch.topk(sim, k=k, largest=True)          # top in subselection
        results: List[Tuple[str, float]] = []
        for sub_i, v in zip(rel.tolist(), vals.tolist()):
            if v >= self.confidence_threshold:
                global_idx = idxs[sub_i]
                results.append((self._attr_all_texts[global_idx], float(v)))
        return results