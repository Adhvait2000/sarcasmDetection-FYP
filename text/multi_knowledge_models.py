"""
Multi-Knowledge Fusion Models
Implements weighted attention mechanisms for combining captions, ANPs, and attributes
"""
import numpy as np
import torch    
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import List, Dict, Tuple, Optional
import math

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
        # Ensure inputs are on the same device as the model
        dev = next(self.bert_model.parameters()).device

        # move text dict to device
        text_input = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in text_input.items()}

        # move knowledge dicts to device (if any)
        if knowledge_inputs is not None:
            ki_moved = []
            for kd in knowledge_inputs:
                if kd is None:
                    ki_moved.append(None)
                else:
                    ki_moved.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kd.items()})
            knowledge_inputs = ki_moved
        # -------- Text branch --------
        text_output = self.bert_model(**text_input)[0]         # (B, Lt+2, 768)
        text_output = text_output[:, 1:-1, :]                  # remove [CLS], [SEP] -> (B, Lt, 768)

        text_embeddings = self.text_projection(text_output)    # (B, Lt, D)

        text_scores = self.importance_scorer(text_embeddings).squeeze(-1)  # (B, Lt)

        # NEW: safe softmax
        logits = text_scores * lam
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)  # replace NaN with 0
        row_sums = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(row_sums == 0, torch.full_like(probs, 1.0 / probs.size(-1)), probs)

        text_scores = probs


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
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_knowledge_types = num_knowledge_types

        # Temperature and Epsilon
        self.tau = 2.0              # Start high to flatten weights
        self.epsilon_floor = 0.05 


        self.entropy_lambda = 5e-3  # Conservative weight
        self.register_buffer("target_entropy", torch.tensor(np.log(num_knowledge_types)))
        
        # Cross-attention (text <- knowledge) and (knowledge <- text)
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)

        # GLOBAL priors (learned, like your current param) — optional but helpful
        self.global_type_logit = nn.Parameter(torch.zeros(num_knowledge_types))

        # NEW: per-sample gating network over per-type knowledge, conditioned on pooled text
        self.text_pool = nn.AdaptiveAvgPool1d(1)  # simple [B,L,D] -> [B,D]
        self.type_gate = nn.Sequential(
            nn.Linear(2*input_size, input_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size//2, 1)           # scalar score per type
        )

        # Output projections + residual norms
        self.out_proj_text = nn.Sequential(
            nn.Linear(input_size, input_size), nn.Dropout(dropout)
        )
        self.out_proj_know = nn.Sequential(
            nn.Linear(input_size, input_size), nn.Dropout(dropout)
        )
        self.ln_text = nn.LayerNorm(input_size)
        self.ln_know = nn.LayerNorm(input_size)

    def _per_type_mask(self, knowledge_masks, B, K, device):
        # Build [B,K] mask (True=ignore) from optional list/tensor
        if knowledge_masks is None:
            return torch.zeros(B, K, dtype=torch.bool, device=device)
        if isinstance(knowledge_masks, list):
            cols = []
            for m in knowledge_masks:
                if m is None:
                    cols.append(torch.ones(B, 1, dtype=torch.bool, device=device))
                else:
                    mb = m.to(device)
                    if mb.dim() == 3 and mb.size(-1) == 1:  # [B,L,1] -> [B,L]
                        mb = mb.squeeze(-1)
                    all_pad = mb.bool().all(dim=1, keepdim=True)  # [B,1]
                    cols.append(all_pad)
            kp = torch.cat(cols, dim=1)  # [B,K]
            return kp[:, :K] if kp.size(1) >= K else torch.cat(
                [kp, torch.zeros(B, K-kp.size(1), dtype=torch.bool, device=device)], dim=1
            )
        if torch.is_tensor(knowledge_masks) and knowledge_masks.dim() == 2:
            return knowledge_masks.to(device).bool()
        return torch.zeros(B, K, dtype=torch.bool, device=device)

    def forward(self, text_embeddings, knowledge_embeddings, knowledge_masks=None,
                knowledge_scores: torch.Tensor = None):
        """
        Returns:
            text_out: [B, L, D]
            know_out: [B, K, D]
            per_sample_weights: [B, K]
            entropy_loss: scalar (0 in eval or when K_active<=1)
        """
        B, L, D = text_embeddings.shape
        _, K, Dk = knowledge_embeddings.shape
        assert D == Dk, "Dim mismatch."
        device = text_embeddings.device

        # --- Build per-type mask (True = masked/inactive) ---
        if knowledge_masks is not None:
            kp_mask = self._per_type_mask(knowledge_masks, B, K, device)
        else:
            kp_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        active = (~kp_mask).float()                # [B, K] 1 = active, 0 = masked
        K_act = active.sum(dim=1, keepdim=True)    # [B, 1]
        single_active = (K_act.squeeze(1) <= 1)    # [B] bool

        entropy_loss = torch.tensor(0.0, device=device)

        if K == 1:
            # Trivial uniform
            per_sample_weights = active  # already 1 for the only stream
            weighted_knowledge = knowledge_embeddings  # * 1
        else:
            # ----- Compute logits only once -----
            t_pool = self.text_pool(text_embeddings.transpose(1, 2)).squeeze(-1)  # [B, D]
            t_rep = t_pool.unsqueeze(1).expand(B, K, D)                           # [B, K, D]
            gate_inp = torch.cat([t_rep, knowledge_embeddings], dim=-1)           # [B, K, 2D]
            type_logits_local = self.type_gate(gate_inp).squeeze(-1)              # [B, K]

            logits = type_logits_local + self.global_type_logit.view(1, K)        # [B, K]
            if knowledge_scores is not None:
                ks = knowledge_scores.to(device)
                ks = (ks - ks.mean(dim=1, keepdim=True)) / (ks.std(dim=1, keepdim=True) + 1e-6)
                logits = logits + ks

            # mask out inactive streams
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(kp_mask, neg_inf)
            all_inf = torch.isneginf(logits).all(dim=1, keepdim=True)
            logits = torch.where(all_inf, torch.zeros_like(logits), logits)

            # softmax with temperature
            w = F.softmax(logits / self.tau, dim=1)                               # [B, K]

            # ---- ε-floor over active streams only ----
            eps = getattr(self, "epsilon_floor", 0.0)
            if eps > 0:
                # add eps to active entries only, then renormalize over active mass
                w = w * active + eps * active
                w = w / (w.sum(1, keepdim=True) + eps * K_act)

            # ---- Per-sample override for K_active <= 1 (fast-path) ----
            if single_active.any():
                # set weight exactly uniform over the active stream(s) for those samples
                w_single = active[single_active]
                w_single = w_single / w_single.sum(1, keepdim=True).clamp_min(1.0)
                w = w.clone()
                w[single_active] = w_single

            per_sample_weights = w

            # ---- Entropy regularization only when K_active > 1 ----
            if self.training and getattr(self, "entropy_lambda", 0.0) > 0.0:
                w_active = per_sample_weights * active
                w_active = w_active / (w_active.sum(1, keepdim=True) + 1e-8)
                ent = -(w_active.clamp_min(1e-8) * w_active.clamp_min(1e-8).log()).sum(1)
                mask_multi = (K_act.squeeze(1) > 1).float()
                target = torch.log(K_act.squeeze(1).clamp(min=1.0))
                entropy_loss = self.entropy_lambda * ((ent - target) ** 2 * mask_multi).mean()

            weighted_knowledge = knowledge_embeddings * per_sample_weights.unsqueeze(-1)

        # ===== Cross-attention blocks (same as yours) =====
        kp_mask_fix = kp_mask.clone()
        all_masked = kp_mask_fix.all(dim=1)
        if all_masked.any():
            idx = all_masked.nonzero(as_tuple=False).squeeze(-1)
            kp_mask_fix[idx, 0] = False

        att_text, _ = self.attention(
            query=text_embeddings, key=weighted_knowledge, value=weighted_knowledge,
            key_padding_mask=kp_mask_fix
        )
        text_out = self.ln_text(text_embeddings + self.out_proj_text(torch.nan_to_num(att_text)))

        att_know, _ = self.attention(
            query=weighted_knowledge, key=text_embeddings, value=text_embeddings
            # key_padding_mask not needed for text keys
        )
        know_out = self.ln_know(weighted_knowledge + self.out_proj_know(torch.nan_to_num(att_know)))

        # NEW: zero out masked knowledge positions in the output (safer residual)
        know_out = know_out.masked_fill(kp_mask.unsqueeze(-1), 0.0)

        return text_out, know_out, per_sample_weights, entropy_loss

