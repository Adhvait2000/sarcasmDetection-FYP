"""
Multi-Knowledge Fusion Models
Implements weighted attention mechanisms for combining captions, ANPs, and attributes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import List, Dict, Tuple, Optional

class MultiKnowledgeFusion(nn.Module):
    def __init__(self, input_size=768, output_size=300, num_knowledge_types=3, 
                 attention_heads=8, dropout=0.1):
        """
        Multi-knowledge fusion module with weighted attention
        
        Args:
            input_size: Input dimension size
            output_size: Output dimension size
            num_knowledge_types: Number of knowledge types (caption, ANP, attribute)
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiKnowledgeFusion, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_knowledge_types = num_knowledge_types
        self.attention_heads = attention_heads
        
        # Knowledge type embeddings
        self.knowledge_type_embeddings = nn.Embedding(num_knowledge_types, input_size)
        
        # Multi-head attention for knowledge fusion
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Knowledge importance scoring
        self.knowledge_scorer = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, 1)
        )
        
        # Knowledge type gate
        self.knowledge_gate = nn.Sequential(
            nn.Linear(input_size * num_knowledge_types, num_knowledge_types),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_size * num_knowledge_types, output_size),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, knowledge_embeddings: List[torch.Tensor], 
                knowledge_masks: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-knowledge fusion
        
        Args:
            knowledge_embeddings: List of knowledge embeddings [caption, anp, attribute]
            knowledge_masks: List of knowledge masks (optional)
            
        Returns:
            fused_knowledge: Fused knowledge representation
            knowledge_weights: Knowledge type importance weights
        """
        batch_size = knowledge_embeddings[0].size(0)
        max_length = max(emb.size(1) for emb in knowledge_embeddings)
        
        # Pad all knowledge embeddings to same length
        padded_embeddings = []
        for emb in knowledge_embeddings:
            if emb.size(1) < max_length:
                padding = torch.zeros(batch_size, max_length - emb.size(1), 
                                   emb.size(2), device=emb.device)
                padded_emb = torch.cat([emb, padding], dim=1)
            else:
                padded_emb = emb
            padded_embeddings.append(padded_emb)
        
        # Add knowledge type embeddings
        type_enhanced_embeddings = []
        for i, emb in enumerate(padded_embeddings):
            type_emb = self.knowledge_type_embeddings(
                torch.full((batch_size, max_length), i, device=emb.device, dtype=torch.long)
            )
            enhanced_emb = emb + type_emb
            type_enhanced_embeddings.append(enhanced_emb)
        
        # Concatenate all knowledge types
        all_knowledge = torch.cat(type_enhanced_embeddings, dim=1)  # (B, L*K, D)
        
        # Apply self-attention across all knowledge
        attended_knowledge, attention_weights = self.knowledge_attention(
            all_knowledge, all_knowledge, all_knowledge
        )
        
        # Reshape to separate knowledge types
        attended_knowledge = attended_knowledge.reshape(batch_size, max_length, 
                                                       self.num_knowledge_types, self.input_size)
        
        # Compute knowledge importance scores
        knowledge_scores = []
        for i in range(self.num_knowledge_types):
            scores = self.knowledge_scorer(attended_knowledge[:, :, i, :])
            knowledge_scores.append(scores)
        
        knowledge_scores = torch.cat(knowledge_scores, dim=-1)  # (B, L, K)
        
        # Apply knowledge gate
        knowledge_weights = self.knowledge_gate(
            attended_knowledge.reshape(batch_size, max_length, -1)
        )  # (B, L, K)
        
        # Weighted combination
        weighted_knowledge = torch.sum(
            attended_knowledge * knowledge_weights.unsqueeze(-1), dim=2
        )  # (B, L, D)
        
        # Project to output size
        fused_knowledge = self.output_projection(
            attended_knowledge.reshape(batch_size, max_length, -1)
        )
        
        return fused_knowledge, knowledge_weights

class EnhancedTextEncoder(nn.Module):
    def __init__(self, input_size=768, output_size=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, dropout=0.1):
        """
        Enhanced text encoder with multi-knowledge support
        
        Args:
            input_size: Input dimension size
            output_size: Output dimension size
            knowledge_types: List of knowledge types to use
            max_knowledge_length: Maximum length for knowledge sequences
            dropout: Dropout rate
        """
        super(EnhancedTextEncoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        
        # BERT model for text encoding
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Knowledge fusion module
        self.knowledge_fusion = MultiKnowledgeFusion(
            input_size=input_size,
            output_size=output_size,
            num_knowledge_types=len(knowledge_types),
            dropout=dropout
        )
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout)
        )
        
        # Knowledge projection
        self.knowledge_projection = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout)
        )
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(output_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout)
        )
        
        # Importance scoring
        self.importance_scorer = nn.Linear(output_size, 1)
        
    def forward(self, text_input, knowledge_inputs=None, knowledge_masks=None, lam=1):
        # -------- Text branch --------
        text_output = self.bert_model(**text_input)[0]         # (B, Lt+2, 768)
        text_output = text_output[:, 1:-1, :]                  # remove [CLS], [SEP] -> (B, Lt, 768)

        text_embeddings = self.text_projection(text_output)    # (B, Lt, D)
        text_scores = self.importance_scorer(text_embeddings).squeeze(-1)  # (B, Lt)
        text_scores = F.softmax(text_scores * lam, dim=-1)

        # -------- Knowledge branch --------
        knowledge_embeddings = None
        knowledge_scores = None

        if knowledge_inputs is not None:
            # Encode each knowledge type with the same BERT
            knowledge_list = []
            for knowledge_input in knowledge_inputs:
                if knowledge_input is not None:
                    k_out = self.bert_model(**knowledge_input)[0]   # (B, Lk+2, 768)
                    k_out = k_out[:, 1:-1, :]                       # (B, Lk, 768)
                    knowledge_list.append(k_out)
                else:
                    # Fallback: zero-length => create a single zero token then pad later
                    # To keep shapes simple, create a (B, 1, 768) zero tensor
                    dummy = torch.zeros(text_output.size(0), 1, self.input_size, device=text_output.device, dtype=text_output.dtype)
                    knowledge_list.append(dummy)

            # Fuse across all knowledge tokens to get contextualized per-token knowledge reprs (B, L, 768)
            fused_knowledge, knowledge_weights = self.knowledge_fusion(knowledge_list, knowledge_masks)
            # fused_knowledge: (B, L, 768) ; knowledge_weights: (B, L, K)

            # --- Option A pooling: turn (B, L, D) + (B, L, K) -> (B, K, D) ---
            # First project fused tokens to D (to match downstream attention dimension)
            fused_knowledge_proj = self.knowledge_projection(fused_knowledge)  # (B, L, D)

            # Weighted average per knowledge type over sequence length
            # type_embeddings[b,k,d] = sum_l fused_knowledge_proj[b,l,d] * weights[b,l,k] / sum_l weights[b,l,k]
            eps = 1e-8
            # einsum: (B,L,D) x (B,L,K) -> (B,K,D)
            type_embeddings = torch.einsum('bld,blk->bkd', fused_knowledge_proj, knowledge_weights)
            denom = knowledge_weights.sum(dim=1, keepdim=False).unsqueeze(-1) + eps  # (B, K, 1)
            type_embeddings = type_embeddings / denom                                # (B, K, D)

            # Per-type scores (optional; useful for diagnostics or further gating)
            knowledge_embeddings = type_embeddings                                    # (B, K, D)
            knowledge_scores = self.importance_scorer(knowledge_embeddings).squeeze(-1)  # (B, K)
            knowledge_scores = F.softmax(knowledge_scores * lam, dim=-1)

        return text_embeddings, text_scores, knowledge_embeddings, knowledge_scores


class WeightedKnowledgeAttention(nn.Module):
    def __init__(self, input_size=768, num_heads=8, dropout=0.1, num_knowledge_types=3):
        """
        Weighted attention mechanism for knowledge integration
        
        Args:
            input_size: Input dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_knowledge_types: Number of knowledge types to handle
        """
        super(WeightedKnowledgeAttention, self).__init__()
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_knowledge_types = num_knowledge_types
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Knowledge type weighting
        self.knowledge_weights = nn.Parameter(torch.ones(num_knowledge_types))
        self.knowledge_softmax = nn.Softmax(dim=0)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_embeddings, knowledge_embeddings, knowledge_masks=None):
        """
        Forward pass for weighted knowledge attention
        
        Args:
            text_embeddings: Text embeddings (B, L, D)
            knowledge_embeddings: Knowledge embeddings (B, K, D)
            knowledge_masks: Knowledge masks (optional)
            
        Returns:
            attended_text: Text with knowledge attention
            attended_knowledge: Knowledge with text attention
            attention_weights: Attention weights
        """
        # Compute knowledge type weights
        knowledge_type_weights = self.knowledge_softmax(self.knowledge_weights)
        
        # Apply knowledge type weighting
        weighted_knowledge = knowledge_embeddings * knowledge_type_weights.unsqueeze(0).unsqueeze(-1)
        
        # Cross-attention between text and knowledge
        attended_text, text_attention = self.attention(
            text_embeddings, weighted_knowledge, weighted_knowledge,
            key_padding_mask=knowledge_masks
        )
        
        attended_knowledge, knowledge_attention = self.attention(
            weighted_knowledge, text_embeddings, text_embeddings
        )
        
        # Project outputs
        attended_text = self.output_projection(attended_text)
        attended_knowledge = self.output_projection(attended_knowledge)
        
        return attended_text, attended_knowledge, text_attention