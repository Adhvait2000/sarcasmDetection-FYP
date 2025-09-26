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

class EnhancedBaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, 
                 img_path=None, knowledge_types=[0], max_knowledge_length=20,
                 confidence_threshold=0.7, frequency_threshold=2, dataset_percentage=100.0):
        """
        Enhanced dataset class supporting multiple knowledge types
        
        Args:
            type: "train","val","test"
            max_length: the max_length for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding
            use_np: whether use noun phrase as relation matching node
            knowledge_types: List of types [0=no knowledge, 1=caption, 2=ANP, 3=attribute, 4=hybrid]
            max_knowledge_length: max tokens per knowledge channel
            confidence_threshold: minimum confidence for knowledge acceptance
            frequency_threshold: minimum frequency for knowledge acceptance
            dataset_percentage: proportion of dataset to use (1.0 = 100%)
        """
        self.type = type
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.use_np = use_np
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.dataset_percentage = dataset_percentage
        
        # Initialize knowledge extractor and filter
        self.knowledge_extractor = KnowledgeExtractor(confidence_threshold=confidence_threshold)
        
        # Load dataset
        with open(self.text_path) as f:
            self.full_dataset = json.load(f)
        self.full_img_set = torch.load(self.img_path)
        
        # Apply dataset sampling
        self.dataset, self.img_set = self._sample_dataset()
        print(f"Dataset sampling: Using {len(self.dataset)} samples ({self.dataset_percentage*100:.1f}% of {len(self.full_dataset)} total samples)")
        
        # Pre-extract knowledge if needed
        self.extracted_knowledge = self._pre_extract_knowledge()

    def _sample_dataset(self) -> Tuple[List, torch.Tensor]:
        """
        Sample a percentage of the dataset

        Returns:
            Tuple of (sampled_dataset, sampled_img_set)
        """
        total_samples = len(self.full_dataset)
        if self.dataset_percentage >= 1.0:
            return self.full_dataset, self.full_img_set

        labels = [int(sample[2]) for sample in self.full_dataset]
        unique_labels, _ = np.unique(labels, return_counts=True)

        sampled_dataset = []
        sampled_img_set = []

        # Seed once for reproducibility
        np.random.seed(42)
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            # ensure >=1 per label when possible; cap at available
            label_target = max(1, int(len(label_indices) * self.dataset_percentage))
            label_target = min(label_target, len(label_indices))
            if label_target == 0:
                continue
            selected_indices = np.random.choice(label_indices, size=label_target, replace=False)
            for idx in selected_indices:
                sampled_dataset.append(self.full_dataset[idx])
                sampled_img_set.append(self.full_img_set[idx])

        combined = list(zip(sampled_dataset, sampled_img_set))
        if not combined:
            # fallback: use entire dataset to avoid zip(*) error
            return self.full_dataset, self.full_img_set

        np.random.shuffle(combined)
        sampled_dataset, sampled_img_set = zip(*combined)
        return list(sampled_dataset), torch.stack(sampled_img_set)


    def _pre_extract_knowledge(self) -> Dict[int, Dict]:
        """
        Pre-extract knowledge for all samples to avoid repeated computation.
        Extract both ANPs and attributes first (using raw ANPs as optional context),
        then filter and trim each stream independently.
        """
        def _filter_and_trim(pairs, max_len):
            if not pairs:
                return []
            return [k for k, conf in pairs][:max_len]

        extracted_knowledge = {}

        for idx, _ in enumerate(self.dataset):
            knowledge_dict = {}

            # Only extract if ANY knowledge type needs it
            needs_anps = 2 in self.knowledge_types or 4 in self.knowledge_types
            needs_attrs = 3 in self.knowledge_types or 4 in self.knowledge_types
            
            if needs_anps or needs_attrs:
                img = self.img_set[idx]
                
                # Extract ANPs ONCE if needed
                anps_with_conf = []
                if needs_anps:
                    anps_with_conf = self.knowledge_extractor.extract_anps_with_clip(img)
                
                # Extract attributes (using raw ANPs for context if available)
                attrs_with_conf = []
                if needs_attrs:
                    raw_anp_terms = [a for a, c in anps_with_conf] if anps_with_conf else []
                    attrs_with_conf = self.knowledge_extractor.extract_attributes(img, raw_anp_terms)
                
                # Filter and trim
                knowledge_dict['anps'] = _filter_and_trim(anps_with_conf, self.max_knowledge_length) if needs_anps else []
                knowledge_dict['attributes'] = _filter_and_trim(attrs_with_conf, self.max_knowledge_length) if needs_attrs else []
            else:
                knowledge_dict['anps'] = []
                knowledge_dict['attributes'] = []
                
            extracted_knowledge[idx] = knowledge_dict

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
            kd = {}
            ex = self.extracted_knowledge.get(index, {})

            # Captions (pseudo-caption from text, trimmed)
            if 1 in self.knowledge_types:
                kd["caption"] = twitter[: self.max_knowledge_length]

            if 2 in self.knowledge_types:
                kd["anps"] = ex.get("anps", [])

            if 3 in self.knowledge_types:
                kd["attributes"] = ex.get("attributes", [])

            if 4 in self.knowledge_types:
                # Hybrid: Combine ANPs and attributes (deduplicated, order-preserving)
                seen = set()
                hybrid = []
                for item in ex.get("anps", []):
                    if item not in seen:
                        seen.add(item); hybrid.append(item)
                for item in ex.get("attributes", []):
                    if item not in seen:
                        seen.add(item); hybrid.append(item)
                kd["hybrid"] = hybrid[: self.max_knowledge_length]
            
            if kd:
                knowledge_data = kd

        # Return with or without knowledge
        if knowledge_data is None:
            return img, twitter, dep, label
        else:
            return img, twitter, dep, label, knowledge_data

    def __len__(self):
        """Returns length of the dataset"""
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