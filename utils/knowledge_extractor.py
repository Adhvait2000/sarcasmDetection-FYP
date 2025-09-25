import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Optional, Dict
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
        """
        self.device = torch.device(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.confidence_threshold = confidence_threshold

        # Predefined ANP templates for better extraction
        self.anp_templates = [
            "a {} person", "a {} object", "a {} scene", "a {} animal",
            "a {} building", "a {} vehicle", "a {} food", "a {} plant"
        ]

        # Common attributes for filtering
        self.emotional_attributes = [
            "happy", "sad", "angry", "surprised", "confused", "excited",
            "bored", "worried", "proud", "embarrassed", "amused", "frustrated"
        ]
        self.stylistic_attributes = [
            "bright", "dark", "colorful", "monochrome", "blurry", "sharp",
            "modern", "vintage", "casual", "formal", "messy", "clean"
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

    def _generate_anp_candidates(self) -> List[str]:
        """Generate candidate ANPs using templates and common nouns."""
        candidates = []
        common_nouns = [
            "person", "man", "woman", "child", "baby", "boy", "girl",
            "car", "truck", "bike", "bus", "train", "plane",
            "house", "building", "office", "school", "hospital",
            "dog", "cat", "bird", "fish", "horse", "cow",
            "tree", "flower", "grass", "mountain", "river", "ocean",
            "food", "pizza", "burger", "cake", "fruit", "vegetable",
            "phone", "computer", "book", "chair", "table", "bed"
        ]
        for template in self.anp_templates:
            for noun in common_nouns:
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
        attrs = self.emotional_attributes + self.stylistic_attributes

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

    def extract_anps_with_clip(self, image, max_anps=10) -> List[Tuple[str, float]]:
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

        k = min(max_anps, sim.numel())
        vals, idxs = torch.topk(sim, k=k, largest=True)
        out = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            if v >= self.confidence_threshold:
                out.append((candidate_anps[i], float(v)))
        return out

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
