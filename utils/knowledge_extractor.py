"""
Knowledge Extraction and Filtering Module
Handles ANP and attribute extraction with confidence-based filtering
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Tuple
import json

class KnowledgeExtractor:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", confidence_threshold=0.7):
        """
        Initialize knowledge extractor with CLIP for ANP extraction
        
        Args:
            clip_model_name: CLIP model to use
            confidence_threshold: Minimum confidence for ANP/attribute acceptance
        """
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
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

    def extract_anps_with_clip(self, image, max_anps=10) -> List[Tuple[str, float]]:
        """
        Extract ANPs using CLIP with confidence scores
        
        Args:
            image: PIL Image or tensor
            max_anps: Maximum number of ANPs to extract
            
        Returns:
            List of (anp, confidence) tuples
        """
        # Convert image to PIL if needed
        if torch.is_tensor(image):
            # Convert tensor to PIL (assuming normalized tensor)
            image = self._tensor_to_pil(image)
        
        # Prepare image for CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Generate candidate ANPs
        candidate_anps = self._generate_anp_candidates()
        
        # Get CLIP predictions
        text_inputs = self.clip_processor(text=candidate_anps, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (image_features @ text_features.T).squeeze()
            
            # Get top ANPs
            top_indices = torch.argsort(similarity, descending=True)[:max_anps]
            top_anps = [(candidate_anps[i], similarity[i].item()) for i in top_indices]
            
            # Filter by confidence threshold
            filtered_anps = [(anp, conf) for anp, conf in top_anps if conf > self.confidence_threshold]
            
        return filtered_anps

    def extract_attributes(self, image, anps: List[str]) -> List[Tuple[str, float]]:
        """
        Extract attributes based on detected ANPs
        
        Args:
            image: PIL Image or tensor
            anps: List of detected ANPs
            
        Returns:
            List of (attribute, confidence) tuples
        """
        # Combine emotional and stylistic attributes
        all_attributes = self.emotional_attributes + self.stylistic_attributes
        
        # Create attribute candidates for each ANP
        attribute_candidates = []
        for anp in anps:
            for attr in all_attributes:
                attribute_candidates.append(f"{attr} {anp}")
        
        # Use CLIP to score attributes
        inputs = self.clip_processor(images=image, return_tensors="pt")
        text_inputs = self.clip_processor(text=attribute_candidates, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Normalize and compute similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze()
            
            # Extract top attributes
            top_indices = torch.argsort(similarity, descending=True)[:20]
            top_attributes = [(attribute_candidates[i], similarity[i].item()) for i in top_indices]
            
            # Filter by confidence
            filtered_attributes = [(attr, conf) for attr, conf in top_attributes if conf > self.confidence_threshold]
            
        return filtered_attributes

    def filter_knowledge_by_frequency(self, knowledge_list: List[Tuple[str, float]], 
                                    min_frequency=2) -> List[Tuple[str, float]]:
        """
        Filter knowledge by frequency threshold
        
        Args:
            knowledge_list: List of (knowledge, confidence) tuples
            min_frequency: Minimum frequency for inclusion
            
        Returns:
            Filtered knowledge list
        """
        # Count frequencies
        frequency_dict = {}
        for knowledge, conf in knowledge_list:
            if knowledge in frequency_dict:
                frequency_dict[knowledge] += 1
            else:
                frequency_dict[knowledge] = 1
        
        # Filter by frequency
        filtered_knowledge = []
        for knowledge, conf in knowledge_list:
            if frequency_dict[knowledge] >= min_frequency:
                filtered_knowledge.append((knowledge, conf))
        
        return filtered_knowledge

    def _generate_anp_candidates(self) -> List[str]:
        """Generate candidate ANPs using templates and common nouns"""
        candidates = []
        
        # Common nouns for ANP generation
        common_nouns = [
            "person", "man", "woman", "child", "baby", "boy", "girl",
            "car", "truck", "bike", "bus", "train", "plane",
            "house", "building", "office", "school", "hospital",
            "dog", "cat", "bird", "fish", "horse", "cow",
            "tree", "flower", "grass", "mountain", "river", "ocean",
            "food", "pizza", "burger", "cake", "fruit", "vegetable",
            "phone", "computer", "book", "chair", "table", "bed"
        ]
        
        # Generate ANPs using templates
        for template in self.anp_templates:
            for noun in common_nouns:
                candidates.append(template.format(noun))
        
        return candidates

    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image (placeholder - implement based on your image format)"""
        # This is a placeholder - implement based on your image format
        # You might need to denormalize and convert to PIL
        return tensor

class KnowledgeFilter:
    def __init__(self, confidence_threshold=0.7, frequency_threshold=2):
        """
        Knowledge filtering module
        
        Args:
            confidence_threshold: Minimum confidence for knowledge acceptance
            frequency_threshold: Minimum frequency for knowledge acceptance
        """
        self.confidence_threshold = confidence_threshold
        self.frequency_threshold = frequency_threshold
        
    def filter_knowledge(self, knowledge_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Apply confidence and frequency filtering
        
        Args:
            knowledge_list: List of (knowledge, confidence) tuples
            
        Returns:
            Filtered knowledge list
        """
        # Filter by confidence
        confidence_filtered = [
            (knowledge, conf) for knowledge, conf in knowledge_list 
            if conf >= self.confidence_threshold
        ]
        
        # Filter by frequency (if frequency data is available)
        # This would require tracking frequencies across the dataset
        frequency_filtered = self._apply_frequency_filter(confidence_filtered)
        
        return frequency_filtered
    
    def _apply_frequency_filter(self, knowledge_list):
        """Apply frequency-based filtering"""
        # This is a placeholder - implement frequency tracking
        # You might want to maintain a global frequency counter
        return knowledge_list