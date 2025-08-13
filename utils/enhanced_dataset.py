"""
Enhanced Dataset Class for Multi-Knowledge Support
Supports captions, ANPs, and attributes with proper handling
"""
import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Tuple, Optional
from utils.knowledge_extractor import KnowledgeExtractor, KnowledgeFilter

class EnhancedBaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, 
                 img_path=None, knowledge_types=[0], max_knowledge_length=20,
                 confidence_threshold=0.7, frequency_threshold=2):
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
        """
        self.type = type
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.use_np = use_np
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        
        # Initialize knowledge extractor and filter
        self.knowledge_extractor = KnowledgeExtractor(confidence_threshold=confidence_threshold)
        self.knowledge_filter = KnowledgeFilter(confidence_threshold, frequency_threshold)
        
        # Load dataset
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.img_set = torch.load(self.img_path)
        
        # Pre-extract knowledge if needed
        self.extracted_knowledge = self._pre_extract_knowledge()

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

    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Returns:
            img: (49, 768) Tensor
            text_emb: (token_len, 768) Tensor
            text_seq: (word_len) List
            dep: List
            word_len: Int
            token_len: Int
            label: Int
            knowledge_data: Dict (if knowledge_types > 0)
        """
        sample = self.dataset[index]

        # Handle different dataset types
        if self.type == "train":
            label = sample[2]
            text = sample[3]
        else:
            label = sample[3]
            text = sample[4]

        # Get text data
        if self.use_np:
            twitter = text["chunk_cap"]
            dep = text["chunk_dep"]
            chunk_index = text["chunk_index"]
        else:
            twitter = text["token_cap"]
            dep = text["token_dep"]

        img = self.img_set[index]
        
        # Handle knowledge data
        knowledge_data = None
        if any(k > 0 for k in self.knowledge_types):
            knowledge_data = self._prepare_knowledge_data(index, sample)
        
        if knowledge_data is None:
            return img, twitter, dep, label
        else:
            return img, twitter, dep, label, knowledge_data

    def _prepare_knowledge_data(self, index: int, sample: List) -> Dict:
        """
        Prepare knowledge data for the sample
        
        Args:
            index: Sample index
            sample: Sample data
            
        Returns:
            Dictionary containing knowledge data
        """
        knowledge_data = {}
        
        # Handle different knowledge types
        if 1 in self.knowledge_types:  # Caption
            if len(sample) > 4:
                knowledge_data['caption'] = sample[4]  # Assuming caption is at index 4
        
        if 2 in self.knowledge_types or 3 in self.knowledge_types:  # ANP or Attribute
            if index in self.extracted_knowledge:
                extracted = self.extracted_knowledge[index]
                if 2 in self.knowledge_types:
                    knowledge_data['anps'] = extracted['anps']
                if 3 in self.knowledge_types:
                    knowledge_data['attributes'] = extracted['attributes']
        
        if 4 in self.knowledge_types:  # Hybrid (all knowledge types)
            # Combine all knowledge types
            combined_knowledge = []
            
            # Add caption if available
            if len(sample) > 4:
                combined_knowledge.extend(sample[4])
            
            # Add extracted ANPs and attributes
            if index in self.extracted_knowledge:
                extracted = self.extracted_knowledge[index]
                combined_knowledge.extend(extracted['anps'])
                combined_knowledge.extend(extracted['attributes'])
            
            knowledge_data['hybrid'] = combined_knowledge
        
        return knowledge_data

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

    def pad_collate(self, batch):
        """
        Enhanced padding collate function
        
        Returns:
            Enhanced batch with multiple knowledge types
        """
        # Extract basic data
        xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        xs = torch.stack(xs)
        
        twitters = list(map(lambda t: t[self.twitter_dim], batch))
        token_lens = [len(twitter) for twitter in twitters]
        
        # Handle knowledge data
        knowledge_data = None
        if any(k > 0 for k in self.knowledge_types):
            knowledge_data = self._process_knowledge_data(batch)
        
        # Process text data (similar to original)
        encoded_cap = self.tokenizer(twitters, is_split_into_words=True, 
                                    return_tensors="pt", truncation=True,
                                    max_length=100, padding=True)
        
        # Create word spans and masks
        word_spans = []
        word_len = []
        for index_encode, len_token in enumerate(token_lens):
            word_span_ = []
            for i in range(len_token):
                word_span = encoded_cap[index_encode].word_to_tokens(i)
                if word_span is not None:
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            word_spans.append(word_span_)
            word_len.append(len(word_span_))
        
        max_len1 = max(word_len)
        mask_batch1 = self._construct_mask_text(word_len, max_len1)
        
        # Process dependencies
        deps1 = [x[self.dep_dim] for x in batch]
        deps1_ = []
        for dep in deps1:
            deps1_.append([d for d in dep if d[0] < max_len1 and d[1] < max_len1])
        
        # Create edge data
        from utils.data_utils import construct_edge_text
        edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(
            deps=deps1_, max_length=max_len1, use_np=self.use_np
        )
        
        # Create labels
        labels = torch.tensor(list(map(lambda t: t[self.label_dim], batch)), dtype=torch.long)
        
        # Return basic data
        result = [xs, encoded_cap, word_spans, word_len, mask_batch1, 
                 edge_cap1, gnn_mask_1, np_mask_1, labels]
        
        # Add knowledge data if available
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