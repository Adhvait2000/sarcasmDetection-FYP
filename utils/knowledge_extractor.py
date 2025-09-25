import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
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
        Initialize knowledge extractor with CLIP for ANP extraction
        """
        self.device = torch.device(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()  # inference-only
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

        # ===== CACHES =====
        # Cache ANP candidate strings and their encoded text features on device
        self._cached_anp_strings: Optional[List[str]] = None
        self._cached_anp_text_features: Optional[torch.Tensor] = None  # [N, D] on device

        # Attribute text encode cache (via single-text CPU cache)
        self._attr_encode_cache_size = 4096

    def _autocast(self):
        # Use AMP only on CUDA; on CPU use a no-op context.
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()

    def _generate_anp_candidates(self) -> List[str]:
        """Generate candidate ANPs using templates and common nouns"""
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

    def _ensure_cached_anp_text_features(self):
        """Encode ANP candidates once and cache their text features on the correct device."""
        if self._cached_anp_text_features is not None:
            return
        self._cached_anp_strings = self._generate_anp_candidates()
        text_inputs = self.clip_processor(
            text=self._cached_anp_strings, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.inference_mode():
            with self._autocast():
                text_features = self.clip_model.get_text_features(**text_inputs)  # [N, D]
        # Normalize once
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._cached_anp_text_features = text_features  # keep on device

    @lru_cache(maxsize=4096)
    def _encode_text_cached_cpu(self, text: str) -> Tuple[Tuple[float, ...],]:
        """
        Encode a single text prompt on CPU as a tuple of floats (hashable for lru_cache).
        Later we convert to a torch tensor on the correct device.
        """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.inference_mode():
            # Temporarily run on CPU to get a small vector; then return to original device
            model_cpu = self.clip_model.to("cpu")
            feats = model_cpu.get_text_features(**inputs)  # [1, D]
            self.clip_model.to(self.device)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return (tuple(feats.squeeze(0).float().cpu().numpy().tolist()),)

    def _encode_texts_to_device_tensor(self, texts: List[str]) -> torch.Tensor:
        """Encode many texts using the single-text cache; returns [N, D] tensor on device."""
        rows = []
        for t in texts:
            (vec_tuple,) = self._encode_text_cached_cpu(t)
            rows.append(vec_tuple)
        arr = torch.tensor(rows, dtype=torch.float32, device=self.device)  # already normalized
        return arr

    def extract_anps_with_clip(self, image, max_anps=10) -> List[Tuple[str, float]]:
        """
        Extract ANPs using CLIP with confidence scores
        """
        if torch.is_tensor(image):
            image = self._tensor_to_pil(image)
            if image is None:
                return []

        # Ensure cached ANP text features exist on device
        self._ensure_cached_anp_text_features()
        candidate_anps = self._cached_anp_strings
        text_features = self._cached_anp_text_features  # [N, D] on device

        # Image -> features
        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            with self._autocast():
                image_features = self.clip_model.get_image_features(**img_inputs)  # [1, D]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Similarity: [1, D] @ [D, N] -> [1, N]
        similarity = (image_features @ text_features.T).squeeze(0)  # [N]

        k = min(max_anps, similarity.size(0))
        topk = torch.topk(similarity, k=k, largest=True)
        top_indices = topk.indices.tolist()
        top_scores = topk.values.tolist()

        results = []
        for idx, score in zip(top_indices, top_scores):
            if score >= self.confidence_threshold:
                results.append((candidate_anps[idx], float(score)))
        return results

    def extract_attributes(self, image, anps: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract attributes based on detected ANPs.
        Builds attribute prompt strings, encodes (cached), computes similarity to image.
        """
        if not anps:
            return []

        # Handle tensor images
        if torch.is_tensor(image):
            image = self._tensor_to_pil(image)
            if image is None:
                return []

        # Combine emotional and stylistic attributes
        all_attributes = self.emotional_attributes + self.stylistic_attributes

        # Create attribute candidates for each ANP
        attribute_candidates: List[str] = []
        for anp in anps:
            for attr in all_attributes:
                attribute_candidates.append(f"{attr} {anp}")

        # Encode image
        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            with self._autocast():
                image_features = self.clip_model.get_image_features(**img_inputs)  # [1, D]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [1, D]

        # Encode attribute texts (cached; normalized)
        text_features = self._encode_texts_to_device_tensor(attribute_candidates)  # [M, D]

        # Similarity: [1, D] @ [D, M] -> [1, M]
        similarity = (image_features @ text_features.T).squeeze(0)  # [M]

        k = min(top_k, similarity.size(0))
        topk = torch.topk(similarity, k=k, largest=True)
        top_indices = topk.indices.tolist()
        top_scores = topk.values.tolist()

        results = []
        for idx, score in zip(top_indices, top_scores):
            if score >= self.confidence_threshold:
                results.append((attribute_candidates[idx], float(score)))
        return results

    def _tensor_to_pil(self, tensor) -> Optional[Image.Image]:
        """
        Convert torch Tensor to PIL Image.
        Accepts:
          - [C, H, W] float in [0,1] or [0,255]
          - [H, W, C] float in [0,1] or [0,255]
          - uint8 variants
        Returns None if unknown layout.
        """
        if not torch.is_tensor(tensor):
            return None

        t = tensor.detach().cpu()
        if t.ndim == 3:
            if t.shape[0] in (1, 3):  # [C, H, W]
                t = t.permute(1, 2, 0)
        elif t.ndim != 3:
            return None  # unsupported

        if t.dtype == torch.float32 or t.dtype == torch.float16 or t.dtype == torch.float64:
            arr = t.numpy()
            # Heuristic scale
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        elif t.dtype == torch.uint8:
            arr = t.numpy()
        else:
            arr = t.to(torch.uint8).numpy()

        # Ensure 3 channels
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

        return Image.fromarray(arr, mode="RGB")


class KnowledgeFilter:
    def __init__(self, confidence_threshold=0.7, frequency_threshold=2):
        """
        Knowledge filtering module
        """
        self.confidence_threshold = confidence_threshold
        self.frequency_threshold = frequency_threshold

    def filter_knowledge(self, knowledge_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Apply confidence and (optional) frequency filtering
        """
        # Filter by confidence
        confidence_filtered = [
            (knowledge, conf) for knowledge, conf in knowledge_list
            if conf >= self.confidence_threshold
        ]
        # Frequency filter placeholder — pass-through for now
        return confidence_filtered
