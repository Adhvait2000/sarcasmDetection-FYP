"""
Enhanced Dataset Class for Multi-Knowledge Support
Supports captions, ANPs, and attributes with proper handling
"""
import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Tuple, Optional
from utils.knowledge_extractor import KnowledgeExtractor
import numpy as np

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

class EnhancedBaseSet(Dataset):
    def __init__(
        self,
        type: str = "train",
        max_length: int = 100,
        text_path: Optional[str] = None,
        use_np: bool = False,
        img_path: Optional[str] = None,
        knowledge_types: List[int] = [0],
        max_knowledge_length: int = 20,
        dataset_percentage: float = 1.0,       # PROPORTION (1.0 = 100%)
        anp_attr_cache_path: Optional[str] = "/caches/anp_attr_all.jsonl",  
        caption_cache_path: Optional[str] = "/caches/captions_all.jsonl",  
    ):
        """
        Enhanced dataset class supporting multiple knowledge types from cached files

        knowledge_types: [0=no knowledge, 1=caption, 2=ANP, 3=attribute, 4=hybrid]
        """
        self.type = type
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.use_np = use_np
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.dataset_percentage = float(dataset_percentage)

        # Load dataset
        with open(self.text_path, "r") as f:
            self.full_dataset = json.load(f)
        self.full_img_set = torch.load(self.img_path)

        # Load cached knowledge files
        self.anp_attr_cache = self._load_jsonl_cache(anp_attr_cache_path)
        self.caption_cache = self._load_jsonl_cache(caption_cache_path)

        # Apply dataset sampling
        self.dataset, self.img_set = self._sample_dataset()
        print(
            f"Dataset sampling: Using {len(self.dataset)} samples "
            f"({self.dataset_percentage * 100:.1f}% of {len(self.full_dataset)} total samples)"
        )

        # Pre-extract knowledge from cache
        self.extracted_knowledge = self._pre_extract_knowledge_from_cache()

    def _load_jsonl_cache(self, cache_path):
        """Load JSONL cache file into a dictionary keyed by image_id"""
        cache_dict = {}
        if not cache_path or not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        with open(cache_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                image_id = str(entry.get("image_id", "")).strip()
                if image_id:
                    cache_dict[image_id] = entry
        return cache_dict


    def _sample_dataset(self) -> Tuple[List, torch.Tensor]:
        """
        Sample a percentage of the dataset (class-balanced).
        Returns:
            Tuple of (sampled_dataset, sampled_img_set)
        """
        if self.dataset_percentage >= 1.0:
            return self.full_dataset, self.full_img_set

        labels = [int(sample[2]) for sample in self.full_dataset]
        unique_labels, _ = np.unique(labels, return_counts=True)

        sampled_dataset = []
        sampled_img_set = []

        np.random.seed(42)
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            target = max(1, int(len(label_indices) * self.dataset_percentage))
            target = min(target, len(label_indices))
            if target == 0:
                continue
            selected_indices = np.random.choice(label_indices, size=target, replace=False)
            for idx in selected_indices:
                sampled_dataset.append(self.full_dataset[idx])
                sampled_img_set.append(self.full_img_set[idx])

        combined = list(zip(sampled_dataset, sampled_img_set))
        if not combined:
            return self.full_dataset, self.full_img_set

        np.random.shuffle(combined)
        sampled_dataset, sampled_img_set = zip(*combined)
        return list(sampled_dataset), torch.stack(sampled_img_set)

    def _pre_extract_knowledge_from_cache(self) -> Dict[int, Dict]:
        """
        Extract knowledge from cached JSONL files for all samples.
        Maps dataset entries to cached knowledge by image_id.
        """
        def _dedup_trim(seq, k):
            if not seq:
                return []
            # order-preserving de-dup
            seen, out = set(), []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
                if len(out) >= k:
                    break
            return out

        extracted_knowledge: Dict[int, Dict] = {}

        for idx, sample in enumerate(self.dataset):
            kd = {"anps": [], "attributes": [], "caption": []}

            # Expect sample[0] to be image_id; fallback to index if not present
            image_id = str(sample[0]) if len(sample) > 0 else str(idx)

            # ANPs/Attributes
            cached_aa = self.anp_attr_cache.get(image_id, {})
            if 2 in self.knowledge_types:
                kd["anps"] = _dedup_trim(cached_aa.get("anps", []), self.max_knowledge_length)
            if 3 in self.knowledge_types:
                kd["attributes"] = _dedup_trim(cached_aa.get("attributes", []), self.max_knowledge_length)

            # Captions
            cached_cap = self.caption_cache.get(image_id, {})
            if 1 in self.knowledge_types:
                cap_text = str(cached_cap.get("caption", "")).strip()
                kd["caption"] = cap_text.split()[: self.max_knowledge_length]

            extracted_knowledge[idx] = kd

        return extracted_knowledge

    def __getitem__(self, index):
        """
        Returns (img, twitter_tokens, dep_edges, label) and (optionally) knowledge_data.
        """
        sample = self.dataset[index]

        # Parse fields from your JSON format
        text_str = sample[1]
        label = int(sample[2])

        # Tokenize for collate function
        twitter = text_str.split()

        # No dependency edges in your JSON
        dep = []

        # Image embedding
        img = self.img_set[index]

        # Build knowledge data based on requested types
        knowledge_data = None
        if any(k > 0 for k in self.knowledge_types):
            ex = self.extracted_knowledge.get(index, {})
            kd = {}

            if 1 in self.knowledge_types:
                kd["caption"] = ex.get("caption", [])
            if 2 in self.knowledge_types:
                kd["anps"] = ex.get("anps", [])
            if 3 in self.knowledge_types:
                kd["attributes"] = ex.get("attributes", [])
            if 4 in self.knowledge_types:
                # Hybrid: ANPs + Attributes (order-preserving de-dup)
                seen, hybrid = set(), []
                for item in ex.get("anps", []):
                    if item not in seen:
                        seen.add(item); hybrid.append(item)
                for item in ex.get("attributes", []):
                    if item not in seen:
                        seen.add(item); hybrid.append(item)
                kd["hybrid"] = hybrid[: self.max_knowledge_length]

            if kd:
                knowledge_data = kd

        return (img, twitter, dep, label) if knowledge_data is None else (img, twitter, dep, label, knowledge_data)

    def __len__(self):
        return len(self.dataset)


class MultiKnowledgePadCollate:
    def __init__(self, img_dim=0, twitter_dim=1, dep_dim=2, label_dim=3, 
                 knowledge_dim=4, use_np=False, max_knowledge_length=20,
                 knowledge_types=[0], text_max_length=100):
        """
        Enhanced padding collate function for multiple knowledge types
        
        Args:
            img_dim: dimension for image data
            twitter_dim: dimension for text data
            dep_dim: dimension for dependency data
            label_dim: dimension for label data
            knowledge_dim: dimension for knowledge data
            use_np: whether use noun phrase
            max_knowledge_length: maximum length for knowledge sequences
            knowledge_types: list of knowledge types to handle
        """
        self.img_dim = img_dim
        self.twitter_dim = twitter_dim
        self.dep_dim = dep_dim
        self.label_dim = label_dim
        self.knowledge_dim = knowledge_dim
        self.use_np = use_np
        self.max_knowledge_length = max_knowledge_length
        self.knowledge_types = knowledge_types
        self.text_max_length = text_max_length
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        
    #     return result
    def pad_collate(self, batch):
        # --- images ---
        xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        xs = torch.stack(xs)

        # --- texts as list-of-words (you already return tokens in __getitem__) ---
        twitters = [t[self.twitter_dim] for t in batch]   # List[List[str]]
        token_lens = [len(tw) if isinstance(tw, list) else 0 for tw in twitters]

        # --- knowledge (only if requested AND present) ---
        has_knowledge = any(
            isinstance(t, (list, tuple)) and len(t) > self.knowledge_dim and t[self.knowledge_dim] is not None
            for t in batch
        )
        knowledge_data = None
        if any(k > 0 for k in self.knowledge_types) and has_knowledge:
            knowledge_data = self._process_knowledge_data(batch)

        # --- tokenize text ---
        # Ensure we always pass a list of tokens (or empty) for each sample
        safe_twitters = [tw if isinstance(tw, list) else [] for tw in twitters]
        encoded_cap = self.tokenizer(
            safe_twitters,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.text_max_length,
            padding=True
        )

        # --- word spans & lengths (robust) ---
        # Use word_ids API (works with fast tokenizers); fall back to simple contiguous spans
        word_spans = []
        word_len = []
        try:
            for bi, tw in enumerate(safe_twitters):
                ids = encoded_cap.word_ids(batch_index=bi)
                spans = []
                if ids is not None:
                    # build [start_token, end_token] per original word index
                    current = {}
                    for ti, wid in enumerate(ids):
                        if wid is None:
                            continue
                        if wid not in current:
                            current[wid] = [ti, ti]  # start,end
                        else:
                            current[wid][1] = ti
                    # preserve order 0..len(tw)-1, clip to tokenizer’s max
                    for wid in range(len(tw)):
                        if wid in current:
                            s, e = current[wid]
                            spans.append([s, e])
                word_spans.append(spans)
                word_len.append(len(spans))
        except Exception:
            # very safe fallback: treat words as single tokens by index
            for bi, tw in enumerate(safe_twitters):
                spans = [[i, i] for i in range(min(len(tw), 100))]
                word_spans.append(spans)
                word_len.append(len(spans))

        max_len1 = max(word_len) if word_len else 0
        mask_batch1 = self._construct_mask_text(word_len, max_len1) if max_len1 > 0 \
                    else torch.zeros(len(batch), 0, dtype=torch.bool)

        # --- deps cleanup ---
        deps1 = [t[self.dep_dim] if (isinstance(t, (list, tuple)) and len(t) > self.dep_dim and t[self.dep_dim] is not None) else [] for t in batch]
        deps1_ = []
        for dep in deps1:
            dep = dep or []
            # keep only edges within current max_len1
            deps1_.append([d for d in dep if isinstance(d, (list, tuple)) and len(d) == 2 and
                        0 <= d[0] < max_len1 and 0 <= d[1] < max_len1])

        # --- edge construction (defensive) ---
        from utils.data_utils import construct_edge_text
        if max_len1 == 0:
            # empty text → empty graphs/masks
            edge_cap1 = torch.zeros((len(batch), 2, 0), dtype=torch.long)
            gnn_mask_1 = torch.ones((len(batch), 0), dtype=torch.bool)
            np_mask_1 = torch.zeros((len(batch), 0), dtype=torch.bool)
        else:
            try:
                edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(
                    deps=deps1_,
                    max_length=max_len1,
                    use_np=self.use_np  # keep whatever you set in constructor (False)
                )
            except Exception as e:
                # fallback to empty edges if upstream expects chunks or non-empty deps
                edge_cap1 = torch.zeros((len(batch), 2, 0), dtype=torch.long)
                gnn_mask_1 = torch.ones((len(batch), 0), dtype=torch.bool)
                np_mask_1 = torch.zeros((len(batch), 0), dtype=torch.bool)

        # --- labels ---
        labels = torch.tensor([t[self.label_dim] for t in batch], dtype=torch.long)

        # --- pack result ---
        result = [xs, encoded_cap, word_spans, word_len, mask_batch1,
                edge_cap1, gnn_mask_1, np_mask_1, labels]
        if knowledge_data is not None:
            result.extend(knowledge_data)
        return result

    def _process_knowledge_data(self, batch):
        """
        Process knowledge data for the batch
        
        Args:
            batch: Batch data
            
        Returns:
            List of processed knowledge data
        """
        knowledge_items = [x[self.knowledge_dim] for x in batch if x[self.knowledge_dim] is not None]
        
        if not knowledge_items:
            return None
        
        # Process different knowledge types
        processed_knowledge = []
        
        for knowledge_type in self.knowledge_types:
            if knowledge_type == 0:
                continue
                
            # Extract knowledge for this type
            type_knowledge = []
            for item in knowledge_items:
                if knowledge_type == 1 and 'caption' in item:
                    type_knowledge.append(item['caption'])
                elif knowledge_type == 2 and 'anps' in item:
                    type_knowledge.append(item['anps'])
                elif knowledge_type == 3 and 'attributes' in item:
                    type_knowledge.append(item['attributes'])
                elif knowledge_type == 4 and 'hybrid' in item:
                    type_knowledge.append(item['hybrid'])
                else:
                    type_knowledge.append([])
            
            # Tokenize and pad knowledge
            if any(len(k) > 0 for k in type_knowledge):
                encoded_know = self.tokenizer(type_knowledge, is_split_into_words=True,
                                            return_tensors="pt", truncation=True,
                                            max_length=self.max_knowledge_length, padding=True)
                
                # Create knowledge word spans
                know_word_spans = self._create_knowledge_word_spans(encoded_know, type_knowledge)
                
                # Create knowledge masks
                know_lens = [len(k) for k in type_knowledge]
                mask_batch_know = self._construct_mask_text(know_lens, self.max_knowledge_length)
                
                processed_knowledge.extend([encoded_know, know_word_spans, mask_batch_know])
            else:
                # Empty knowledge
                processed_knowledge.extend([None, None, None])
        
        return processed_knowledge

    def _create_knowledge_word_spans(self, encoded_know, knowledge_list):
        """Create word spans for knowledge tokens using HF fast tokenizer word_ids API."""
        spans_all = []
        for bi, words in enumerate(knowledge_list):
            spans = []
            try:
                ids = encoded_know.word_ids(batch_index=bi)
            except Exception:
                ids = None

            if ids is not None:
                current = {}
                for ti, wid in enumerate(ids):
                    if wid is None:
                        continue
                    if wid not in current:
                        current[wid] = [ti, ti]
                    else:
                        current[wid][1] = ti

                max_len = min(len(words), self.max_knowledge_length)
                for wid in range(max_len):
                    if wid in current:
                        s, e = current[wid]
                        spans.append([s, e])
            else:
                # Safe fallback: treat words as single-token spans by index
                max_len = min(len(words), self.max_knowledge_length)
                spans = [[i, i] for i in range(max_len)]

            spans_all.append(spans)
        return spans_all


    def _construct_mask_text(self, seq_len, max_length):
        """Construct mask for text sequences"""
        mask = torch.zeros(len(seq_len), max_length, dtype=torch.bool)
        for i, length in enumerate(seq_len):
            if length < max_length:
                mask[i, length:] = True
        return mask

    def __call__(self, batch):
        return self.pad_collate(batch)