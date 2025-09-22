"""
Enhanced Multi-Knowledge Fusion Models
Implements sophisticated tri-modal fusion with knowledge-guided attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import List, Dict, Tuple, Optional
import math

class TriModalFusion(nn.Module):
    """
    Sophisticated tri-modal fusion replacing naive decision-level concatenation
    """
    def __init__(self, hidden_dim=300, num_heads=8, num_layers=3, dropout=0.1):
        super(TriModalFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Three-way cross-attention modules
        self.text_img_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_know_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.img_know_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Modality-specific projections to ensure consistent dimensions
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.img_projection = nn.Linear(hidden_dim, hidden_dim)
        self.know_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion transformer layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=min(num_heads, 6),  # Cap at 6 heads
                dim_feedforward=int(1.5 * hidden_dim),  # Reduced from 2x to 1.5x
                dropout=dropout,
                batch_first=True
            ) for _ in range(min(num_layers, 2))  # Cap at 2 layers
        ])
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(3, hidden_dim)  # text, image, knowledge
        
        # Output attention pooling
        self.output_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_feats, img_feats, know_feats, 
            text_mask=None, img_mask=None, know_mask=None):
        """
        Args:
            text_feats: [B, Lt, D] 
            img_feats: [B, Li, D]
            know_feats: [B, Lk, D]
            masks: Optional attention masks for each modality
        """
        B = text_feats.size(0)
        device = text_feats.device
        
        # Project to consistent dimensions
        text_proj = self.text_projection(text_feats)
        img_proj = self.img_projection(img_feats)
        know_proj = self.know_projection(know_feats)
        
        # Add modality type embeddings
        text_type = self.modality_embeddings(torch.zeros(B, text_proj.size(1), device=device, dtype=torch.long))
        img_type = self.modality_embeddings(torch.ones(B, img_proj.size(1), device=device, dtype=torch.long))
        know_type = self.modality_embeddings(torch.full((B, know_proj.size(1)), 2, device=device, dtype=torch.long))
        
        text_proj = text_proj + text_type
        img_proj = img_proj + img_type
        know_proj = know_proj + know_type
        
        # Create tri-modal sequence
        tri_modal = torch.cat([text_proj, img_proj, know_proj], dim=1)  # [B, Lt+Li+Lk, D]
        
        # Create combined attention mask using the actual tensor dimensions
        combined_mask = None
        if any(mask is not None for mask in [text_mask, img_mask, know_mask]):
            Lt, Li, Lk = text_proj.size(1), img_proj.size(1), know_proj.size(1)
            
            # Create masks for each modality, defaulting to False (not masked)
            if text_mask is not None:
                text_m = text_mask.to(device).bool()
            else:
                text_m = torch.zeros(B, Lt, device=device, dtype=torch.bool)
                
            if img_mask is not None:
                img_m = img_mask.to(device).bool()
            else:
                img_m = torch.zeros(B, Li, device=device, dtype=torch.bool)
                
            if know_mask is not None:
                know_m = know_mask.to(device).bool()
            else:
                know_m = torch.zeros(B, Lk, device=device, dtype=torch.bool)
            
            # Combine masks to match tri_modal sequence length
            combined_mask = torch.cat([text_m, img_m, know_m], dim=1)
        
        # Apply fusion transformer layers
        for layer in self.fusion_layers:
            tri_modal = layer(tri_modal, src_key_padding_mask=combined_mask)
        
        # Global attention pooling for final representation
        pooled, attn_weights = self.output_attn(
            tri_modal.mean(dim=1, keepdim=True),  # query: global average
            tri_modal,  # key & value: all tokens
            tri_modal,
            key_padding_mask=combined_mask
        )
        
        # Final projection
        fused_repr = self.output_projection(pooled.squeeze(1))  # [B, D]
        
        return fused_repr, attn_weights

class KnowledgeConditionedImageEncoder(nn.Module):
    """
    Image encoder that is conditioned by knowledge embeddings
    """
    def __init__(self, input_dim=768, inter_dim=500, output_dim=300, 
                 knowledge_dim=300, num_heads=8, dropout=0.1):
        super(KnowledgeConditionedImageEncoder, self).__init__()
        
        # Base image encoder components
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inter_dim, output_dim)
        )
        
        # Knowledge conditioning modules
        self.know_to_patch_attn = nn.MultiheadAttention(
            output_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Patch importance scoring with knowledge conditioning
        self.patch_scorer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 2),  # Bottleneck
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, imgs, knowledge_embeddings=None, lam=1):
        """
        Args:
            imgs: [B, K_patches, input_dim]
            knowledge_embeddings: [B, K_types, knowledge_dim] or None
        """
        # Standard image feature extraction
        img_features = self.feature_extractor(imgs)  # [B, K_patches, output_dim]
        
        if knowledge_embeddings is not None:
            # Knowledge-conditioned patch features
            conditioned_features, knowledge_attn = self.know_to_patch_attn(
                query=img_features,
                key=knowledge_embeddings,
                value=knowledge_embeddings
            )
            
            # Combine original and conditioned features
            combined_features = torch.cat([img_features, conditioned_features], dim=-1)
            
            # Compute knowledge-aware patch importance
            patch_scores = self.patch_scorer(combined_features).squeeze(-1)  # [B, K_patches]
            
            # Apply layer norm to conditioned features
            final_features = self.layer_norm(conditioned_features)
        else:
            # Fallback to standard processing
            combined_features = torch.cat([img_features, img_features], dim=-1)
            patch_scores = self.patch_scorer(combined_features).squeeze(-1)
            final_features = self.layer_norm(img_features)
        
        # Safe softmax for patch importance
        patch_scores = patch_scores * lam
        patch_weights = F.softmax(patch_scores, dim=-1)
        patch_weights = torch.nan_to_num(patch_weights, nan=0.0)
        
        # Ensure weights sum to 1
        weight_sums = patch_weights.sum(dim=-1, keepdim=True)
        patch_weights = torch.where(
            weight_sums == 0,
            torch.full_like(patch_weights, 1.0 / patch_weights.size(-1)),
            patch_weights / weight_sums
        )
        
        return final_features, patch_weights

class CrossTypeKnowledgePooling(nn.Module):
    def __init__(self, hidden_dim=300, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, type_vecs: torch.Tensor, type_scores: Optional[torch.Tensor] = None):
        """
        Args:
            type_vecs: [B, K, D]  per-type knowledge vectors (e.g., caption / ANP / attribute)
            type_scores: Optional [B, K] weights per type.
        Returns:
            pooled:  [B, D]   (weighted or mean-pooled across K)
            refined: [B, K, D] (cross-type self-attended)
        """
        if type_vecs is None or type_vecs.size(1) == 0:
            B = 1 if type_vecs is None else type_vecs.size(0)
            D = 300 if type_vecs is None else type_vecs.size(-1)
            device = None if type_vecs is None else type_vecs.device
            return (torch.zeros(B, D, device=device), torch.zeros(B, 0, D, device=device))

        # Cross-type self-attention over K types
        refined, _ = self.mha(type_vecs, type_vecs, type_vecs)      # [B, K, D]
        refined = self.norm(refined + type_vecs)                     # residual

        if type_scores is not None:
            w = F.softmax(type_scores, dim=-1).unsqueeze(-1)         # [B, K, 1]
            pooled = (w * refined).sum(dim=1)                        # [B, D]
        else:
            pooled = refined.mean(dim=1)                             # [B, D]

        return pooled, refined
class KnowledgeGuidedCrossModal(nn.Module):
    """
    Cross-modal interaction guided by knowledge context
    """
    def __init__(self, input_size=300, nhead=8, dim_feedforward=600, 
                 dropout=0.1, cro_layer=6, type_bmco=1):
        super(KnowledgeGuidedCrossModal, self).__init__()
        
        self.input_size = input_size
        self.nhead = nhead
        self.type_bmco = type_bmco
        
        # Base cross-modal transformer layers
        self.cross_modal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(cro_layer)
        ])
        
        # Knowledge guidance modules
        self.knowledge_guidance = nn.MultiheadAttention(
            input_size, nhead, dropout=dropout, batch_first=True
        )
        
        self.knowledge_gate = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.Sigmoid()
        )
        
        # Modality-specific adaptation
        self.image_adapter = nn.Linear(input_size, input_size)
        self.text_adapter = nn.Linear(input_size, input_size)
        
    def forward(self, images, texts, knowledge_context=None, key_padding_mask=None):
        """
        images: [B, Li, D]
        texts:  [B, Lt, D]
        key_padding_mask: [B, Lt] for text (True = pad)
        """
        B, Li, _ = images.size()
        Lt = texts.size(1)

        # Combine (you can change ordering/strategy by type_bmco if needed)
        combined = torch.cat([images, texts], dim=1)  # [B, Li+Lt, D]

        # --- Build a correct combined mask for ALL branches ---
        if key_padding_mask is not None:
            # Ensure boolean, device, and width==Lt
            text_mask = key_padding_mask.to(images.device).bool()
            if text_mask.size(1) != Lt:
                m = min(Lt, text_mask.size(1))
                fixed = torch.zeros(B, Lt, dtype=torch.bool, device=images.device)
                fixed[:, :m] = text_mask[:, :m]
                text_mask = fixed
            img_mask = torch.zeros(B, Li, dtype=torch.bool, device=images.device)
            combined_mask = torch.cat([img_mask, text_mask], dim=1)  # [B, Li+Lt]
        else:
            combined_mask = None

        # Cross-modal encoder
        attended = combined
        for layer in self.cross_modal_layers:
            attended = layer(attended, src_key_padding_mask=combined_mask)

        # Knowledge-guided refinement
        if knowledge_context is not None:
            knowledge_guided, _ = self.knowledge_guidance(
                query=attended, key=knowledge_context, value=knowledge_context
            )
            gate = self.knowledge_gate(torch.cat([attended, knowledge_guided], dim=-1))
            attended = gate * knowledge_guided + (1 - gate) * attended

        # Split back
        attended_images = self.image_adapter(attended[:, :Li, :])
        attended_texts  = self.text_adapter(attended[:, Li:Li+Lt, :])
        return attended_images, attended_texts


class EnhancedHybridModel(nn.Module):
    """
    Sophisticated hybrid model with tri-modal fusion and knowledge conditioning
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768,
                 img_inter_dim=500, img_out_dim=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, img_gat_layer=2,
                 img_gat_drop=0.5, img_gat_head=2, img_patch=49, lam=1, type_bmco=1):
        super(EnhancedHybridModel, self).__init__()
        
        # Store parameters
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.img_out_dim = img_out_dim
        self.knowledge_types = knowledge_types
        self.lam = lam
        
        # Ensure dimensions match
        if self.img_out_dim != self.txt_out_size:
            self.img_out_dim = self.txt_out_size
        
        # Enhanced text encoder with specialized BERT
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=max_knowledge_length,
            dropout=cro_drop
        )
        
        # Knowledge-conditioned image encoder
        self.img_encoder = KnowledgeConditionedImageEncoder(
            input_dim=img_input_dim,
            inter_dim=img_inter_dim,
            output_dim=self.img_out_dim,
            knowledge_dim=self.txt_out_size,
            dropout=cro_drop
        )
        
        # Cross-type knowledge pooling (expects per-type vectors [B, K, D])
        self.knowledge_pooling = CrossTypeKnowledgePooling(
            hidden_dim=self.txt_out_size,
            dropout=cro_drop
        )
        
        # Knowledge-guided cross-modal interaction
        self.cross_modal_interaction = KnowledgeGuidedCrossModal(
            input_size=self.img_out_dim,
            nhead=cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=cro_drop,
            cro_layer=cro_layers,
            type_bmco=type_bmco
        )
        
        # Tri-modal fusion replacing naive concatenation
        self.tri_modal_fusion = TriModalFusion(
            hidden_dim=self.txt_out_size,
            num_heads=cro_heads,
            dropout=cro_drop
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.txt_out_size, self.txt_out_size // 2),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size // 2, 2)
        )
        
    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,
                txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):
        """
        Enhanced forward pass with sophisticated tri-modal fusion
        """
        # 1) Encode text and per-type knowledge vectors
        texts_encoded, text_scores, knowledge_embeddings, knowledge_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        # texts_encoded:        [B, Lt, D]
        # knowledge_embeddings: [B, K, D]  (per-type)
        # knowledge_scores:     [B, K]

        # 2) Cross-type knowledge pooling
        if knowledge_embeddings is not None:
            # pooled_knowledge_vec: [B, D]  (single vector)
            # refined_knowledge:    [B, K, D] (per-type refined)
            pooled_knowledge_vec, refined_knowledge = self.knowledge_pooling(
                knowledge_embeddings, knowledge_scores
            )
        else:
            pooled_knowledge_vec, refined_knowledge = None, None

        # 3) Knowledge-conditioned image encoding uses the refined per-type matrix
        imgs_encoded, patch_weights = self.img_encoder(
            imgs, knowledge_embeddings=refined_knowledge, lam=self.lam
        )
        # imgs_encoded: [B, Li, D], patch_weights: [B, Li]

        # 4) Knowledge-guided cross-modal interaction
        knowledge_context = refined_knowledge  # [B, K, D] or None
        imgs_attended, texts_attended = self.cross_modal_interaction(
            images=imgs_encoded,
            texts=texts_encoded,
            knowledge_context=knowledge_context,
            key_padding_mask=mask_batch
        )
        # imgs_attended:  [B, Li, D]
        # texts_attended: [B, Lt, D]

        # 5) Prepare single-token summaries per modality for fusion
        text_global = texts_attended.mean(dim=1, keepdim=True)   # [B, 1, D]
        img_global  = imgs_attended.mean(dim=1, keepdim=True)    # [B, 1, D]
        if pooled_knowledge_vec is not None:
            know_global = pooled_knowledge_vec.unsqueeze(1)      # [B, 1, D]
        else:
            # If no knowledge available, use a zero token as placeholder
            know_global = torch.zeros_like(text_global)

        # 6) Tri-modal fusion (sequence of three tokens: T=1, I=1, K=1)
        fused_representation, _ = self.tri_modal_fusion(
            text_feats=text_global,
            img_feats=img_global,
            know_feats=know_global
        )  # [B, D]

        # 7) Final classification
        predictions = self.classifier(fused_representation)       # [B, 2]
        return predictions


# Update the existing EnhancedTextEncoder to support specialized BERT
class EnhancedTextEncoder(nn.Module):
    def __init__(self, input_size=768, output_size=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, dropout=0.1, use_specialized_bert=False):
        super(EnhancedTextEncoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        
        # Main BERT for text
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.knowledge_bert = self.bert_model  # share weights

        # 1) Freeze everything
        for p in self.bert_model.parameters():
            p.requires_grad = False

        # 2) Unfreeze only the last 4 layers + pooler/LayerNorm
        TRAINABLE_KEYS = [
            "encoder.layer.6", "encoder.layer.7", 
            "encoder.layer.8", "encoder.layer.9", 
            "encoder.layer.10","encoder.layer.11",
            "pooler", "LayerNorm"
        ]
        for name, p in self.bert_model.named_parameters():
            if any(k in name for k in TRAINABLE_KEYS):
                p.requires_grad = True

        
       
        # Knowledge fusion module (existing)
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
        
        # Importance scoring
        self.importance_scorer = nn.Linear(output_size, 1)
        
    def forward(self, text_input, knowledge_inputs=None, knowledge_masks=None, lam=1):
        # Move inputs to device
        dev = next(self.bert_model.parameters()).device
        text_input = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in text_input.items()}
        
        if knowledge_inputs is not None:
            ki_moved = []
            for kd in knowledge_inputs:
                if kd is None:
                    ki_moved.append(None)
                else:
                    ki_moved.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kd.items()})
            knowledge_inputs = ki_moved
        
        # Text encoding
        text_output = self.bert_model(**text_input)[0]
        text_output = text_output[:, 1:-1, :]  # Remove [CLS], [SEP]
        text_embeddings = self.text_projection(text_output)
        
        # Text importance scoring with safe softmax
        text_scores = self.importance_scorer(text_embeddings).squeeze(-1)
        logits = text_scores * lam
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        text_scores = torch.where(row_sums == 0, 
                                torch.full_like(probs, 1.0 / probs.size(-1)), 
                                probs)
        
        # Knowledge encoding
        knowledge_embeddings = None
        knowledge_scores = None
        
        if knowledge_inputs is not None:
            knowledge_list = []
            for knowledge_input in knowledge_inputs:
                if knowledge_input is not None:
                    k_out = self.knowledge_bert(**knowledge_input)[0]
                    k_out = k_out[:, 1:-1, :]  # Remove [CLS], [SEP]
                    knowledge_list.append(k_out)
                else:
                    dummy = torch.zeros(text_output.size(0), 1, self.input_size, 
                                      device=text_output.device, dtype=text_output.dtype)
                    knowledge_list.append(dummy)
            
            # Fuse knowledge
            fused_knowledge, knowledge_weights = self.knowledge_fusion(knowledge_list, knowledge_masks)
            knowledge_embeddings_proj = self.knowledge_projection(fused_knowledge)
            
            # Per-type pooling using knowledge weights
            eps = 1e-8
            type_embeddings = torch.einsum('bld,blk->bkd', knowledge_embeddings_proj, knowledge_weights)
            denom = knowledge_weights.sum(dim=1, keepdim=False).unsqueeze(-1) + eps
            knowledge_embeddings = type_embeddings / denom
            
            knowledge_scores = self.importance_scorer(knowledge_embeddings).squeeze(-1)
            knowledge_scores = F.softmax(knowledge_scores * lam, dim=-1)
        
        return text_embeddings, text_scores, knowledge_embeddings, knowledge_scores

# Keep existing MultiKnowledgeFusion class unchanged
class MultiKnowledgeFusion(nn.Module):
    def __init__(self, input_size=768, output_size=300, num_knowledge_types=3, 
                 attention_heads=8, dropout=0.1):
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
        all_knowledge = torch.cat(type_enhanced_embeddings, dim=1)
        
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
        
        knowledge_scores = torch.cat(knowledge_scores, dim=-1)
        
        # Apply knowledge gate
        knowledge_weights = self.knowledge_gate(
            attended_knowledge.reshape(batch_size, max_length, -1)
        )
        
        # Weighted combination
        weighted_knowledge = torch.sum(
            attended_knowledge * knowledge_weights.unsqueeze(-1), dim=2
        )
        
        # Project to output size
        fused_knowledge = self.output_projection(
            attended_knowledge.reshape(batch_size, max_length, -1)
        )
        
        return fused_knowledge, knowledge_weights