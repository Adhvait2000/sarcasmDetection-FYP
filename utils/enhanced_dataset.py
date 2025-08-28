"""
Enhanced Dataset Class for Multi-Knowledge Support
Supports captions, ANPs, and attributes with proper handling
"""
import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Tuple, Optional
from utils.knowledge_extractor import KnowledgeExtractor, KnowledgeFilter
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
            knowledge_types: List of knowledge types [0=no knowledge, 1=caption, 2=ANP, 3=attribute, 4=hybrid]
            max_knowledge_length: maximum length for knowledge sequences
            confidence_threshold: minimum confidence for knowledge acceptance
            frequency_threshold: minimum frequency for knowledge acceptance
            dataset_percentage: Percentage of dataset to use (1.0 = 100%, 0.75 = 75%, etc.)
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
        self.knowledge_filter = KnowledgeFilter(confidence_threshold, frequency_threshold)
        
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
        target_samples = int(total_samples * self.dataset_percentage)
        
        if self.dataset_percentage >= 1.0:
            return self.full_dataset, self.full_img_set
        
        # Use stratified sampling to maintain class balance
        # if self.type == "train":
        #     # For training, we want to maintain class distribution
        #     labels = [sample[2] for sample in self.full_dataset]
        # else:
        #     # For val/test, use the appropriate label index
        #     labels = [sample[3] for sample in self.full_dataset]
        labels = [int(sample[2]) for sample in self.full_dataset]
        
        # Get unique labels and their counts
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        sampled_dataset = []
        sampled_img_set = []
        
        for label in unique_labels:
            # Get indices for this label
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            # Calculate how many samples to take for this label
            label_target = int(len(label_indices) * self.dataset_percentage)
            
            # Randomly sample from this label
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(label_indices, size=label_target, replace=False)
            
            # Add selected samples
            for idx in selected_indices:
                sampled_dataset.append(self.full_dataset[idx])
                sampled_img_set.append(self.full_img_set[idx])
        
        # Shuffle the sampled dataset
        combined = list(zip(sampled_dataset, sampled_img_set))
        np.random.shuffle(combined)
        sampled_dataset, sampled_img_set = zip(*combined)
        
        return list(sampled_dataset), torch.stack(sampled_img_set)

    def _pre_extract_knowledge(self) -> Dict[int, Dict]:
        """
        Pre-extract knowledge for all samples to avoid repeated computation
        
        Returns:
            Dictionary mapping sample index to knowledge dict
        """
        extracted_knowledge = {}
        
        for idx, sample in enumerate(self.dataset):
            if 2 in self.knowledge_types or 3 in self.knowledge_types:  # ANP or attribute
                # Get image for this sample
                img = self.img_set[idx]
                
                # Extract ANPs if needed
                anps = []
                if 2 in self.knowledge_types:
                    anps_with_conf = self.knowledge_extractor.extract_anps_with_clip(img)
                    anps = [anp for anp, conf in anps_with_conf]
                
                # Extract attributes if needed
                attributes = []
                if 3 in self.knowledge_types:
                    attributes_with_conf = self.knowledge_extractor.extract_attributes(img, anps)
                    attributes = [attr for attr, conf in attributes_with_conf]
                
                # Filter knowledge
                all_knowledge = anps + attributes
                filtered_knowledge = self.knowledge_filter.filter_knowledge(
                    [(k, 1.0) for k in all_knowledge]  # Assume confidence 1.0 for pre-extracted
                )
                
                extracted_knowledge[idx] = {
                    'anps': anps,
                    'attributes': attributes,
                    'filtered_knowledge': [k for k, conf in filtered_knowledge]
                }
        
        return extracted_knowledge

    # def __getitem__(self, index):
    #     """
    #     Get a sample from the dataset
        
    #     Returns:
    #         img: (49, 768) Tensor
    #         text_emb: (token_len, 768) Tensor
    #         text_seq: (word_len) List
    #         dep: List
    #         word_len: Int
    #         token_len: Int
    #         label: Int
    #         knowledge_data: Dict (if knowledge_types > 0)
    #     """
    #     sample = self.dataset[index]

    #     # Handle different dataset types
    #     # if self.type == "train":
    #     #     label = sample[2]
    #     #     text = sample[3]
    #     # else:
    #     #     label = sample[3]
    #     #     text = sample[4]
    #     label = int(sample[2])
    #     text_str = sample[1]

    #     # Get text data
    #     if self.use_np:
    #         twitter = text["chunk_cap"]
    #         dep = text["chunk_dep"]
    #         chunk_index = text["chunk_index"]
    #     else:
    #         twitter = text["token_cap"]
    #         dep = text["token_dep"]

    #     img = self.img_set[index]
        
    #     # Handle knowledge data
    #     knowledge_data = None
    #     if any(k > 0 for k in self.knowledge_types):
    #         knowledge_data = self._prepare_knowledge_data(index, sample)
        
    #     if knowledge_data is None:
    #         return img, twitter, dep, label
    #     else:
    #         return img, twitter, dep, label, knowledge_data
    def __getitem__(self, index):
        """
        Returns (img, twitter_tokens, dep_edges, label) and (optionally) knowledge_data.
        - twitter_tokens: List[str] (tokenized from raw text)
        - dep_edges: List[Tuple[int,int]] (empty here; your JSON has no deps)
        - label: int (0/1)
        - knowledge_data: Dict with keys among {'anps','attributes','hybrid'} if enabled
        """
        sample = self.dataset[index]

        # Parse fields from your JSON format: [id_str, text_str, label_int, <unused_int>]
        text_str = sample[1]
        label = int(sample[2])

        # Tokenize simply for the collate (it expects word-level tokens)
        twitter = text_str.split()

        # No dependency edges provided in your JSON → empty list is safe
        dep = []

        # Image embedding aligned with the same sampled index
        img = self.img_set[index]

        # --- Knowledge (no captions present in your JSON) ---
        knowledge_data = None
        if any(k > 0 for k in self.knowledge_types):
            kd = {}

            # ANPs / Attributes come from pre-extraction
            ex = self.extracted_knowledge.get(index)
            if ex is not None:
                if 2 in self.knowledge_types:
                    kd["anps"] = ex.get("anps", [])
                if 3 in self.knowledge_types:
                    kd["attributes"] = ex.get("attributes", [])

                if 4 in self.knowledge_types:
                    combo = []
                    combo.extend(ex.get("anps", []))
                    combo.extend(ex.get("attributes", []))
                    kd["hybrid"] = combo

            # If captions (1) were requested, your JSON doesn't have them → skip
            # (Optionally: kd["caption"] = [] to be explicit.)

            if kd:  # only attach if something was added
                knowledge_data = kd

        # Return with or without knowledge payload based on availability
        if knowledge_data is None:
            return img, twitter, dep, label
        else:
            return img, twitter, dep, label, knowledge_data


    # def _prepare_knowledge_data(self, index: int, sample: List) -> Dict:
    #     """
    #     Prepare knowledge data for the sample
        
    #     Args:
    #         index: Sample index
    #         sample: Sample data
            
    #     Returns:
    #         Dictionary containing knowledge data
    #     """
    #     knowledge_data = {}
        
    #     # Handle different knowledge types
    #     if 1 in self.knowledge_types:  # Caption
    #         if len(sample) > 4:
    #             knowledge_data['caption'] = sample[4]  # Assuming caption is at index 4
        
    #     if 2 in self.knowledge_types or 3 in self.knowledge_types:  # ANP or Attribute
    #         if index in self.extracted_knowledge:
    #             extracted = self.extracted_knowledge[index]
    #             if 2 in self.knowledge_types:
    #                 knowledge_data['anps'] = extracted['anps']
    #             if 3 in self.knowledge_types:
    #                 knowledge_data['attributes'] = extracted['attributes']
        
    #     if 4 in self.knowledge_types:  # Hybrid (all knowledge types)
    #         # Combine all knowledge types
    #         combined_knowledge = []
            
    #         # Add caption if available
    #         if len(sample) > 4:
    #             combined_knowledge.extend(sample[4])
            
    #         # Add extracted ANPs and attributes
    #         if index in self.extracted_knowledge:
    #             extracted = self.extracted_knowledge[index]
    #             combined_knowledge.extend(extracted['anps'])
    #             combined_knowledge.extend(extracted['attributes'])
            
    #         knowledge_data['hybrid'] = combined_knowledge
        
    #     return knowledge_data
    def _prepare_knowledge_data(self, index: int, sample) -> dict | None:
        """
        Build the per-sample knowledge payload.
        Your JSON has no caption field, so we only handle ANPs (2), attributes (3),
        and a hybrid bag (4) that concatenates both.

        Returns:
            dict with any of: {"anps": [...], "attributes": [...], "hybrid": [...]}
            or None if nothing applicable is available.
        """
        # If no knowledge requested, skip early
        if not any(k in (2, 3, 4) for k in self.knowledge_types):
            return None

        kd = {}
        ex = self.extracted_knowledge.get(index)  # from _pre_extract_knowledge()

        if ex is not None:
            # Add ANPs
            if 2 in self.knowledge_types:
                kd["anps"] = ex.get("anps", [])

            # Add attributes
            if 3 in self.knowledge_types:
                kd["attributes"] = ex.get("attributes", [])

            # Hybrid = ANPs + attributes (deduped, order-preserved)
            if 4 in self.knowledge_types:
                combo = []
                for item in ex.get("anps", []):
                    if item not in combo:
                        combo.append(item)
                for item in ex.get("attributes", []):
                    if item not in combo:
                        combo.append(item)
                kd["hybrid"] = combo

        # If captions were requested (1) but your dataset lacks them, we simply omit "caption".
        # If nothing ended up in kd, return None so the collate can ignore it.
        return kd or None


    def __len__(self):
        """Returns length of the dataset"""
        return len(self.dataset)

class MultiKnowledgePadCollate:
    def __init__(self, img_dim=0, twitter_dim=1, dep_dim=2, label_dim=3, 
                 knowledge_dim=4, use_np=False, max_knowledge_length=20,
                 knowledge_types=[0]):
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
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # def pad_collate(self, batch):
    #     """
    #     Enhanced padding collate function
        
    #     Returns:
    #         Enhanced batch with multiple knowledge types
    #     """
    #     # Extract basic data
    #     xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
    #     xs = torch.stack(xs)
        
    #     twitters = list(map(lambda t: t[self.twitter_dim], batch))
    #     token_lens = [len(twitter) for twitter in twitters]
        
    #     # Handle knowledge data
    #     knowledge_data = None
    #     if any(k > 0 for k in self.knowledge_types):
    #         knowledge_data = self._process_knowledge_data(batch)
        
    #     # Process text data (similar to original)
    #     encoded_cap = self.tokenizer(twitters, is_split_into_words=True, 
    #                                 return_tensors="pt", truncation=True,
    #                                 max_length=100, padding=True)
        
    #     # Create word spans and masks
    #     word_spans = []
    #     word_len = []
    #     for index_encode, len_token in enumerate(token_lens):
    #         word_span_ = []
    #         for i in range(len_token):
    #             word_span = encoded_cap[index_encode].word_to_tokens(i)
    #             if word_span is not None:
    #                 word_span_.append([word_span[0] - 1, word_span[1] - 1])
    #         word_spans.append(word_span_)
    #         word_len.append(len(word_span_))
        
    #     max_len1 = max(word_len)
    #     mask_batch1 = self._construct_mask_text(word_len, max_len1)
        
    #     # Process dependencies
    #     deps1 = [x[self.dep_dim] for x in batch]
    #     deps1_ = []
    #     for dep in deps1:
    #         deps1_.append([d for d in dep if d[0] < max_len1 and d[1] < max_len1])
        
    #     # Create edge data
    #     from utils.data_utils import construct_edge_text
    #     edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(
    #         deps=deps1_, max_length=max_len1, use_np=self.use_np
    #     )
        
    #     # Create labels
    #     labels = torch.tensor(list(map(lambda t: t[self.label_dim], batch)), dtype=torch.long)
        
    #     # Return basic data
    #     result = [xs, encoded_cap, word_spans, word_len, mask_batch1, 
    #              edge_cap1, gnn_mask_1, np_mask_1, labels]
        
    #     # Add knowledge data if available
    #     if knowledge_data is not None:
    #         result.extend(knowledge_data)
        
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
            max_length=100,
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
        """Create word spans for knowledge tokens"""
        know_word_spans = []
        for index_encode, knowledge in enumerate(knowledge_list):
            word_span_ = []
            len_token = len(knowledge)
            if len_token > self.max_knowledge_length:
                len_token = self.max_knowledge_length
            for i in range(len_token):
                word_span = encoded_know[index_encode].word_to_tokens(i)
                if word_span is not None:
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            know_word_spans.append(word_span_)
        return know_word_spans

    def _construct_mask_text(self, seq_len, max_length):
        """Construct mask for text sequences"""
        mask = torch.zeros(len(seq_len), max_length, dtype=torch.bool)
        for i, length in enumerate(seq_len):
            if length < max_length:
                mask[i, length:] = True
        return mask

    def __call__(self, batch):
        return self.pad_collate(batch)