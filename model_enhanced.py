"""
Enhanced Model Variants with Sophisticated Fusion
Replaces naive fusion with tri-modal attention and knowledge conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from images.image_models import ImageEncoder
from interraction.inter_models import CroModality
import utils.gat as tg_conv
import math

# Import the enhanced fusion components
from text.multi_knowledge_models import (
    TriModalFusion, 
    KnowledgeConditionedImageEncoder,
    CrossTypeKnowledgePooling,
    KnowledgeGuidedCrossModal,
    EnhancedTextEncoder,
    MultiKnowledgeFusion
)

def pool_tokens_to_words_batch(seq, score, word_spans, pad_len):
    """
    Pool token-level seq [B,T,D] to word-level [B,pad_len,D] using word_spans (list of list of [s,e]).
    Also pools 'score' to [B,pad_len,1] if provided (reduces any channel dim to scalar).
    """
    B, T, D = seq.shape
    device = seq.device

    out_seq = torch.zeros((B, pad_len, D), dtype=seq.dtype, device=device)

    out_score = None
    if score is not None:
        score = score.to(device)
        if score.dim() == 2:         # [B,T]
            score = score.unsqueeze(-1)          # [B,T,1]
        elif score.dim() == 3 and score.size(-1) not in (1, D):
            score = score.mean(dim=-1, keepdim=True)  # [B,T,1]
        out_score = torch.zeros((B, pad_len, 1), dtype=score.dtype, device=device)

    for b in range(B):
        spans = word_spans[b] if isinstance(word_spans[b], list) else []
        Wb = min(len(spans), pad_len)
        for w in range(Wb):
            s, e = spans[w]
            s = max(0, min(int(s), T - 1))
            e = max(0, min(int(e), T - 1))
            if e < s:
                s, e = e, s
            token_slice = seq[b, s:e+1, :]   # [n,D]
            out_seq[b, w, :] = token_slice.mean(dim=0)
            if out_score is not None:
                token_scores = score[b, s:e+1, :]   # [n,1]
                out_score[b, w, :] = token_scores.mean(dim=0)

    return out_seq, out_score

class BaselineModel(nn.Module):
    """
    Baseline Model: Image + Text + Captions (unchanged for comparison)
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768, 
                 img_inter_dim=500, img_out_dim=300, cro_layers=6, cro_heads=5, 
                 cro_drop=0.5, txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2,
                 img_gat_layer=2, img_gat_drop=0.5, img_gat_head=2, img_patch=49,
                 lam=1, type_bmco=1):
        super(BaselineModel, self).__init__()
        
        # Model parameters
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim
        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_patch = img_patch
        self.lam = lam
        self.type_bmco = type_bmco
        
        # Ensure output dimensions match
        if self.img_out_dim != self.txt_out_size:
            self.img_out_dim = self.txt_out_size
        
        # Encoders
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=[1],  # Only captions
            dropout=self.cro_drop
        )
        
        self.img_encoder = ImageEncoder(
            input_dim=self.img_input_dim,
            inter_dim=self.img_inter_dim,
            output_dim=self.img_out_dim
        )
        
        # Cross-modal interaction
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
        )
        
        # Alignment module
        self.alignment = Alignment(
            input_size=self.img_out_dim,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            img_gat_layer=self.img_gat_layer,
            img_gat_drop=self.img_gat_drop,
            img_gat_head=self.img_gat_head,
            lam=self.lam
        )
        
        # Output layer
        self.output_layer = nn.Linear(2 * self.img_patch, 2)
        
    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq, 
                txt_edge_index, gnn_mask, np_mask, knowledge_inputs=None, 
                knowledge_masks=None):
        # Encode images
        imgs, pv = self.img_encoder(imgs, lam=self.lam)
        
        # Encode text and knowledge
        texts, text_scores, knowledge_embeddings, knowledge_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # Cross-modal interaction
        imgs, texts = self.interaction(
            images=imgs,
            texts=texts,
            key_padding_mask=mask_batch
        )

        # Pool token-level to word-level
        texts, text_scores = pool_tokens_to_words_batch(
            seq=texts,
            score=text_scores,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )
        
        # Alignment
        alignment_scores = self.alignment(
            t2=texts,
            v2=imgs,
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=text_scores,
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            img_edge_index=img_edge_index,
            lam=self.lam
        )
        
        # Apply importance weights
        pv = pv.repeat(1, 2)
        weighted_alignment = alignment_scores * pv
        
        # Final prediction
        predictions = self.output_layer(weighted_alignment)
        
        return predictions

class ImageOnlyModel(nn.Module):
    """
    Image-Only ablation (unchanged)
    """
    def __init__(self, img_input_dim=768, img_inter_dim=500, img_out_dim=200, img_patch=49, drop=0.5, lam=1):
        super().__init__()
        self.img_encoder = ImageEncoder(
            input_dim=img_input_dim,
            inter_dim=img_inter_dim,
            output_dim=img_out_dim
        )
        self.drop = nn.Dropout(drop)
        self.fc = nn.Sequential(
            nn.Linear(img_out_dim, img_out_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(img_out_dim, 2)
        )
        self.lam = lam
    
    def forward(self, imgs, **_):
        v2, pv = self.img_encoder(imgs, lam=self.lam)
        w = torch.softmax(pv, dim=1).unsqueeze(1)
        pooled = torch.bmm(w, v2).squeeze(1)
        logits = self.fc(self.drop(pooled))
        return logits

class EnhancedKnowledgeOnlyModel(nn.Module):
    """
    Enhanced Knowledge-Only Model with sophisticated knowledge processing
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, knowledge_types=[2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1):
        super(EnhancedKnowledgeOnlyModel, self).__init__()
        
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.lam = lam
        
        # Encoder (specialized BERT for knowledge)
        self.knowledge_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            dropout=cro_drop,
            use_specialized_bert=True
        )
        
        # Cross-type pooling: returns pooled [B,D] and refined per-type [B,K,D]
        self.knowledge_pooling = CrossTypeKnowledgePooling(
            hidden_dim=self.txt_out_size,
            num_heads=cro_heads,
            dropout=cro_drop
        )
        
        # Light tri-modal fusion (text + two knowledge tokens)
        self.knowledge_text_fusion = TriModalFusion(
            hidden_dim=self.txt_out_size,
            num_heads=cro_heads,
            num_layers=2,
            dropout=cro_drop
        )
        
        # Optional: alignment module kept for compatibility (unused here)
        self.knowledge_alignment = Alignment(
            input_size=self.txt_out_size,
            txt_gat_layer=txt_gat_layer,
            txt_gat_drop=txt_gat_drop,
            txt_gat_head=txt_gat_head,
            lam=self.lam
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.txt_out_size, self.txt_out_size),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size, self.txt_out_size // 2),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size // 2, 2)
        )

    def forward(self, texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, 
                np_mask, knowledge_inputs, knowledge_masks):
        
        # 1) Encode text + per-type knowledge vectors
        #    texts: [B, Lt, D], knowledge_embeddings: [B, K, D], knowledge_scores: [B, K]
        texts, text_scores, knowledge_embeddings, knowledge_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # 2) Cross-type knowledge pooling
        pooled_knowledge_vec = None
        refined_knowledge = None
        if knowledge_embeddings is not None:
            # pooled:  [B, D] (single vector)
            # refined: [B, K, D] (per-type, refined)
            pooled_knowledge_vec, refined_knowledge = self.knowledge_pooling(
                knowledge_embeddings, knowledge_scores
            )
        
        # 3) Pool text tokens to word level (to respect your downstream usage)
        texts_word, text_scores_word = pool_tokens_to_words_batch(
            seq=texts,
            score=text_scores,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )
        
        # 4) Tri-modal fusion setup
        #    - text branch: single token from word-level (mean)
        #    - knowledge branches: two single-token sequences from per-type (or fallbacks)
        text_global = texts_word.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        if refined_knowledge is not None:
            K = refined_knowledge.size(1)
            if K >= 2:
                know1 = refined_knowledge[:, 0:1, :]  # [B,1,D]
                know2 = refined_knowledge[:, 1:2, :]  # [B,1,D]
            else:
                # only one type: duplicate pooled vector (or the single refined type)
                base = refined_knowledge[:, 0:1, :] if K == 1 else pooled_knowledge_vec.unsqueeze(1)
                know1, know2 = base, base
        else:
            # no knowledge available → zeros as safe fallback
            zeros = torch.zeros_like(text_global)
            know1, know2 = zeros, zeros
        
        # 5) Tri-modal fusion (text, know1, know2)
        fused_repr, _ = self.knowledge_text_fusion(
            text_feats=text_global,   # [B,1,D]
            img_feats=know1,          # [B,1,D]
            know_feats=know2          # [B,1,D]
        )  # -> [B, D]
        
        # 6) Classifier
        predictions = self.classifier(fused_repr)  # [B, 2]
        return predictions


class SuperiorHybridModel(nn.Module):
    """
    Sophisticated Hybrid Model with tri-modal fusion and knowledge conditioning
    Replaces naive concatenation with rich multi-modal interactions
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768,
                 img_inter_dim=500, img_out_dim=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, img_gat_layer=2,
                 img_gat_drop=0.5, img_gat_head=2, img_patch=49, lam=1, type_bmco=1):
        super(SuperiorHybridModel, self).__init__()
        
        # Store parameters
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.img_out_dim = img_out_dim
        self.knowledge_types = knowledge_types
        self.lam = lam
        self.img_patch = img_patch
        
        # Ensure dimensions match
        if self.img_out_dim != self.txt_out_size:
            self.img_out_dim = self.txt_out_size
        
        # Specialized learning: separate BERT for knowledge
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=max_knowledge_length,
            dropout=cro_drop,
            use_specialized_bert=True
        )
        
        # Sophisticated knowledge pooling
        self.knowledge_pooling = CrossTypeKnowledgePooling(
            hidden_dim=self.txt_out_size,
            num_heads=cro_heads,
            dropout=cro_drop
        )
        
        # Knowledge-conditioned image encoder (addresses bottleneck #2)
        self.img_encoder = KnowledgeConditionedImageEncoder(
            input_dim=img_input_dim,
            inter_dim=img_inter_dim,
            output_dim=self.img_out_dim,
            knowledge_dim=self.txt_out_size,
            num_heads=cro_heads,
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
        
        # Tri-modal fusion replacing naive decision concatenation (addresses bottleneck #1)
        self.tri_modal_fusion = TriModalFusion(
            hidden_dim=self.txt_out_size,
            num_heads=min(cro_heads, 6),  # Cap heads
            num_layers=2,  # Reduced from 4 to 2
            dropout=cro_drop
        )   
                
        # Confidence-based weighting
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.txt_out_size, self.txt_out_size // 2),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size // 2, 3),  # text, image, knowledge confidence
            nn.Softmax(dim=-1)
        )
        
        # Final sophisticated classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.txt_out_size, self.txt_out_size),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size, self.txt_out_size // 2),
            nn.ReLU(),
            nn.Dropout(cro_drop),
            nn.Linear(self.txt_out_size // 2, 2)
        )
        
    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,
            txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):
        """
        Sophisticated forward pass with tri-modal fusion and knowledge conditioning
        """
        # 1) Enhanced text and knowledge encoding
        texts_encoded, text_scores, knowledge_embeddings, knowledge_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # 2) Cross-type knowledge pooling (FIXED)
        pooled_knowledge_vec = None
        refined_knowledge = None
        if knowledge_embeddings is not None:
            # Direct pooling of per-type vectors [B, K, D] -> [B, D] and [B, K, D]
            pooled_knowledge_vec, refined_knowledge = self.knowledge_pooling(
                knowledge_embeddings, knowledge_scores
            )
        
        # 3) Knowledge-conditioned image encoding (uses refined per-type matrix)
        imgs_encoded, patch_weights = self.img_encoder(
            imgs, knowledge_embeddings=refined_knowledge, lam=self.lam
        )
        
        # 4) Knowledge-guided cross-modal interaction
        imgs_attended, texts_attended = self.cross_modal_interaction(
            images=imgs_encoded,
            texts=texts_encoded,
            knowledge_context=refined_knowledge,  # Use refined per-type matrix
            key_padding_mask=mask_batch
        )
        
        # 5) Prepare representations for tri-modal fusion
        # Use attention-weighted pooling instead of simple averaging
        text_importance = self.confidence_estimator(texts_attended.mean(dim=1))[:, 0:1]
        img_importance = self.confidence_estimator(imgs_attended.mean(dim=1))[:, 1:2]
        
        # Weighted global representations
        text_weighted = torch.sum(
            texts_attended * text_importance.unsqueeze(-1), dim=1, keepdim=True
        )
        img_weighted = torch.sum(
            imgs_attended * patch_weights.unsqueeze(-1), dim=1, keepdim=True
        )
        
        # 6) Tri-modal fusion
        if pooled_knowledge_vec is not None:
            know_importance = self.confidence_estimator(pooled_knowledge_vec)[:, 2:3]
            know_weighted = pooled_knowledge_vec.unsqueeze(1) * know_importance.unsqueeze(-1)  # [B, 1, D]
            
            # Sophisticated tri-modal fusion (addresses bottleneck #1)
            fused_representation, fusion_attn = self.tri_modal_fusion(
                text_feats=text_weighted,   # [B, 1, D]
                img_feats=img_weighted,     # [B, 1, D]
                know_feats=know_weighted    # [B, 1, D]
            )
        else:
            # Fallback to bi-modal fusion
            combined = torch.cat([text_weighted, img_weighted], dim=1)
            fused_representation = combined.mean(dim=1)
        
        # 7) Final classification with rich representation
        predictions = self.classifier(fused_representation)
        
        return predictions

# Keep the existing Alignment class for compatibility
class Alignment(nn.Module):
    """
    Alignment module for computing alignment scores between modalities
    """
    def __init__(self, input_size=300, txt_gat_layer=2, txt_gat_drop=0.2, 
                 txt_gat_head=5, txt_self_loops=False, img_gat_layer=2, 
                 img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, lam=1):
        super(Alignment, self).__init__()
        
        self.input_size = input_size
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.lam = lam
        
        self.txt_conv = nn.ModuleList([
            tg_conv.GATConv(
                in_channels=self.input_size,
                out_channels=self.input_size,
                heads=self.txt_gat_head,
                concat=False,
                dropout=self.txt_gat_drop,
                fill_value="mean",
                add_self_loops=self.txt_self_loops,
                is_text=True
            ) for _ in range(self.txt_gat_layer)
        ])
        
        self.img_conv = nn.ModuleList([
            tg_conv.GATConv(
                in_channels=self.input_size,
                out_channels=self.input_size,
                heads=self.img_gat_head,
                concat=False,
                dropout=self.img_gat_drop,
                fill_value="mean",
                add_self_loops=self.img_self_loops
            ) for _ in range(self.img_gat_layer)
        ])
        
        self.linear1 = nn.Linear(self.input_size, 1)
        self.linear2 = nn.Linear(self.input_size, 1)
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.ReLU()
        
    def forward(self, t2, v2, edge_index, gnn_mask, score, key_padding_mask, 
                np_mask, img_edge_index, lam=1):
        """
        t2: [B, L, D] (WORD-level embeddings)
        v2: [B, K, D]
        score: per-token weights; accepts [B,L], [B,L,1], or [B,L,D]
        """
        dev = t2.device
        B, L, D = t2.shape

        # score -> [B,L,1] or [B,L,D] (broadcastable)
        if score is None:
            score = torch.full((B, L, 1), 1.0 / max(L, 1), device=dev, dtype=t2.dtype)
        else:
            score = score.to(dev)
            if score.dim() == 2:
                score = score.unsqueeze(-1)
            elif score.dim() == 3 and score.size(-1) not in (1, D):
                score = score.mean(dim=-1, keepdim=True)

        # masks to device & safe shapes
        key_padding_mask = (key_padding_mask.to(dev).bool()
                            if key_padding_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=dev))
        if key_padding_mask.size(1) != L:
            kpm = torch.zeros(B, L, dtype=torch.bool, device=dev)
            m = min(L, key_padding_mask.size(1))
            kpm[:, :m] = key_padding_mask[:, :m]
            key_padding_mask = kpm

        # atomic congruity
        q1 = torch.bmm(t2, v2.permute(0, 2, 1)) / math.sqrt(float(D))
        c = torch.sum(score * t2, dim=1, keepdim=True)

        # token importance
        pa_token = self.linear1(t2).squeeze(-1)
        pa_token = pa_token.masked_fill(key_padding_mask, float("-inf"))

        # safe softmax
        logits = pa_token * lam
        probs = F.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(row_sums == 0, torch.full_like(probs, 1.0 / probs.size(-1)), probs)

        pa_token = probs.unsqueeze(2).expand(B, L, v2.size(1))

        # text GAT
        tnp = t2
        for gat in self.txt_conv:
            tnp = self.norm(torch.stack([
                self.relu1(gat(x, ei.to(dev), mask=m.to(dev)))
                for x, ei, m in zip(tnp, edge_index, gnn_mask)
            ], dim=0))

        # image GAT
        v3 = v2
        if isinstance(img_edge_index, (list, tuple)):
            iei_seq = [ei.to(dev) for ei in img_edge_index]
        elif isinstance(img_edge_index, torch.Tensor) and img_edge_index.dim() == 3:
            iei_seq = [img_edge_index[b].to(dev) for b in range(img_edge_index.size(0))]
        else:
            iei_seq = [img_edge_index.to(dev) for _ in range(v3.size(0))]

        for gat in self.img_conv:
            v3 = self.norm(torch.stack([
                self.relu1(gat(x, ei)) for x, ei in zip(v3, iei_seq)
            ], dim=0))

        # compositional congruity
        tnp = torch.cat([tnp, c], dim=1)
        q2 = torch.bmm(tnp, v3.permute(0, 2, 1)) / math.sqrt(float(tnp.size(2)))

        # NP importance
        pa_np = self.linear2(tnp).squeeze(-1)
        if (np_mask is None) or (np_mask.size(1) != tnp.size(1)):
            np_mask = torch.zeros_like(pa_np, dtype=torch.bool, device=dev)
        else:
            np_mask = np_mask.to(dev).bool()

        pa_np = pa_np.masked_fill(np_mask, float("-inf"))

        # safe softmax over (L+1)
        logits = pa_np * lam                                    
        probs  = F.softmax(logits, dim=1)                       
        probs  = torch.nan_to_num(probs, nan=0.0)               
        row_sums = probs.sum(dim=-1, keepdim=True)              
        probs  = torch.where(                                   
            row_sums == 0,
            torch.full_like(probs, 1.0 / probs.size(-1)),
            probs
        )

        pa_np = probs.unsqueeze(2).expand(B, tnp.size(1), v3.size(1))

        a_1 = torch.sum(q1 * pa_token, dim=1)
        a_2 = torch.sum(q2 * pa_np,   dim=1)
        return torch.cat([a_1, a_2], dim=1)