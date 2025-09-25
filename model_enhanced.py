"""
Enhanced Model Variants
Implements the three configurations: Baseline, Knowledge-only, and Hybrid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from interraction.inter_models import CroModality
import utils.gat as tg_conv
import math


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
    Baseline Model: Image + Text + Captions
    Reproduces Liu et al. (2022) with hierarchical congruity modeling
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
        """
        Forward pass for baseline model
        
        Args:
            imgs: Image tensors
            texts: Text input dictionary
            mask_batch: Text padding masks
            img_edge_index: Image edge indices
            t1_word_seq: Word sequences
            txt_edge_index: Text edge indices
            gnn_mask: GNN masks
            np_mask: Noun phrase masks
            knowledge_inputs: Knowledge inputs (captions only)
            knowledge_masks: Knowledge masks
            
        Returns:
            predictions: Model predictions
        """
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

        # ▶︎ Pool token-level to word-level using spans so shapes match masks/graphs
        #    texts: [B,T,D] -> [B, max_words, D]
        texts, text_scores = pool_tokens_to_words_batch(
            seq=texts,
            score=text_scores,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )
        
        # Alignment (now t2 length == mask/graph length)
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
    Image-Only ablation: classify using image encoder + simple pooling head.
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
        # v2: [B, K, D], pv: [B, K] (patch weights) — matches how you use it elsewhere
        v2, pv = self.img_encoder(imgs, lam=self.lam)
        # normalize weights and pool patches -> [B, D]
        w = torch.softmax(pv, dim=1).unsqueeze(1)      # [B, 1, K]
        pooled = torch.bmm(w, v2).squeeze(1)           # [B, D]
        logits = self.fc(self.drop(pooled))            # [B, 2]
        return logits

class KnowledgeOnlyModel(nn.Module):
    """
    Knowledge-Only Model: ANPs + Attributes (no image patches)
    Uses only extracted knowledge without image features
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, knowledge_types=[2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1):
        super(KnowledgeOnlyModel, self).__init__()
        
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.lam = lam
        
        # Encode text + knowledge
        self.knowledge_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            dropout=self.cro_drop
        )
        
        # Fuse different knowledge streams
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=self.txt_out_size,
            num_heads=self.cro_heads,
            dropout=self.cro_drop,
            num_knowledge_types=len(self.knowledge_types)
        )
        
        # Align fused text (word-level) with knowledge (K nodes)
        self.knowledge_alignment = Alignment(
            input_size=self.txt_out_size,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            lam=self.lam
        )
        
        # --- NEW: dimension-agnostic head ---
        # We will reshape [B, 2K] -> [B, 2, K], pool over K -> [B, 2], then classify.
        self.pool_over_k = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(2, 2)

    def forward(self, texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, 
                np_mask, knowledge_inputs, knowledge_masks):
        # Encode text & knowledge (token-level)
        texts, text_scores, knowledge_embeddings, knowledge_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # Fuse knowledge streams:
        # - texts: (B, Lt, D)
        # - knowledge_embeddings: (B, K, D)  ← per-type vectors from EnhancedTextEncoder (Option A)
        # `WeightedKnowledgeAttention` expects (B, K, D), so this is now correct.
        fused_text, fused_knowledge, _ = self.knowledge_fusion(
            text_embeddings=texts,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_masks=None  # token masks no longer match K; drop or build a (B,K) mask if you have one
        )

        # Pool fused_text tokens -> words.
        # IMPORTANT: pass a token-aligned score. `knowledge_scores` is per-type (B,K), so DO NOT use it here.
        fused_text_word, text_scores_word = pool_tokens_to_words_batch(
            seq=fused_text,
            score=text_scores,  # <-- token-level weights from EnhancedTextEncoder
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )


        # Alignment returns [B, 2K] (two maps concatenated along last dim)
        # Create dummy img_edge_index for knowledge-only model (no images)
        batch_size = fused_knowledge.size(0)
        num_knowledge = fused_knowledge.size(1)
        dummy_img_edge_index = torch.zeros((batch_size, 2, 0), dtype=torch.long, device=fused_knowledge.device)
        
        alignment_scores = self.knowledge_alignment(
            t2=fused_text_word,
            v2=fused_knowledge,          # (B, K, D)
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=text_scores_word,      # token/word-aligned weights
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            img_edge_index=dummy_img_edge_index,
            lam=self.lam
        ) # shape: [B, 2K]

        B, twoK = alignment_scores.shape
        if twoK == 0:
            # Edge-case guard (no knowledge nodes): fall back to zeros
            logits = self.classifier(torch.zeros(B, 2, device=alignment_scores.device, dtype=alignment_scores.dtype))
            return logits

        K = twoK // 2
        a1 = alignment_scores[:, :K]   # [B, K]
        a2 = alignment_scores[:, K:]   # [B, K]

        # Stack into [B, 2, K] then pool over K -> [B, 2]
        stacked = torch.stack([a1, a2], dim=1)              # [B, 2, K]
        pooled = self.pool_over_k(stacked).squeeze(-1)      # [B, 2]

        # Final logits
        predictions = self.classifier(pooled)               # [B, 2]
        return predictions


class HybridModel(nn.Module):
    """
    Hybrid Model: Image + Text + Captions + ANPs + Attributes
    Combines all knowledge sources with weighted attention
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768,
                 img_inter_dim=500, img_out_dim=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, img_gat_layer=2,
                 img_gat_drop=0.5, img_gat_head=2, img_patch=49, lam=1, type_bmco=1):
        super(HybridModel, self).__init__()
        
        # Model parameters
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
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
        
        # Enhanced text encoder with all knowledge types
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            dropout=self.cro_drop
        )
        
        # Image encoder
        self.img_encoder = ImageEncoder(
            input_dim=self.img_input_dim,
            inter_dim=self.img_inter_dim,
            output_dim=self.img_out_dim
        )
        
        # Multi-modal interaction
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
        )
        
        # Knowledge fusion
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=self.txt_out_size,
            num_heads=self.cro_heads,
            dropout=self.cro_drop
        )
        
        # Alignment modules
        self.text_image_alignment = Alignment(
            input_size=self.img_out_dim,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            img_gat_layer=self.img_gat_layer,
            img_gat_drop=self.img_gat_drop,
            img_gat_head=self.img_gat_head,
            lam=self.lam
        )
        
        self.knowledge_alignment = Alignment(
            input_size=self.txt_out_size,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            lam=self.lam
        )
        
        # Output layers
        self.text_image_output = nn.Linear(2 * self.img_patch, 2)
        self.knowledge_output = nn.Linear(self.txt_out_size, 2)
        self.final_fusion = nn.Linear(4, 2)  # Combine both outputs
        
    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):
        """
        Forward pass for hybrid model
        
        Args:
            imgs: Image tensors
            texts: Text input dictionary
            mask_batch: Text padding masks
            img_edge_index: Image edge indices
            t1_word_seq: Word sequences
            txt_edge_index: Text edge indices
            gnn_mask: GNN masks
            np_mask: Noun phrase masks
            knowledge_inputs: All knowledge inputs
            knowledge_masks: Knowledge masks
            
        Returns:
            predictions: Model predictions
        """
        # Encode images
        imgs, pv = self.img_encoder(imgs, lam=self.lam)
        
        # Encode text and knowledge
        texts, text_scores, knowledge_embeddings, knowledge_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # Cross-modal interaction (text-image)
        imgs, texts = self.interaction(
            images=imgs,
            texts=texts,
            key_padding_mask=mask_batch
        )

        # ▶︎ Pool tokens -> words for the text branch used in text-image alignment
        texts_word, text_scores_word = pool_tokens_to_words_batch(
            seq=texts,
            score=text_scores,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )

        # Knowledge fusion (still token-level inputs)
        fused_text, fused_knowledge, attention_weights = self.knowledge_fusion(
            text_embeddings=texts,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_masks=knowledge_masks
        )

        # ▶︎ Pool fused_text to words for knowledge alignment
        fused_text_word, fused_text_word_scores = pool_tokens_to_words_batch(
            seq=fused_text,
            score=text_scores,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )

        B = fused_knowledge.size(0)
        dummy_img_edge_index = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_knowledge.device)

        # Text-image alignment
        text_image_alignment = self.text_image_alignment(
            t2=texts_word,
            v2=imgs,
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=text_scores_word,
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            img_edge_index=img_edge_index,
            lam=self.lam
        )

        # Knowledge alignment
        knowledge_alignment = self.knowledge_alignment(
            t2=fused_text_word,
            v2=fused_knowledge,
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=fused_text_word_scores,
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            img_edge_index=dummy_img_edge_index,
            lam=self.lam
        )
        
        # Apply importance weights
        pv = torch.softmax(pv, dim=1)     
        pv = pv.repeat(1, 2)
        weighted_text_image = text_image_alignment * pv
        
        # Generate predictions
        text_image_pred = self.text_image_output(weighted_text_image)

        if fused_knowledge.size(1) == 0:
            knowledge_pred = torch.zeros(
                fused_knowledge.size(0), 2,
                device=fused_knowledge.device,
                dtype=fused_knowledge.dtype
            )
        else:
            # If you have a padding mask per knowledge token with True=pad, you can do a masked mean.
            # Otherwise, a simple mean over tokens is a safe default:
            pooled = fused_knowledge.mean(dim=1)             # [B, txt_out_size]
            knowledge_pred = self.knowledge_output(pooled)   # [B, 2]

        
        # Final fusion
        combined_features = torch.cat([text_image_pred, knowledge_pred], dim=-1)  # [B, 4]
        predictions = self.final_fusion(combined_features)                        # [B, 2] 
        
        return predictions

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

        # --- score -> [B,L,1] or [B,L,D] (broadcastable) ---
        if score is None:
            score = torch.full((B, L, 1), 1.0 / max(L, 1), device=dev, dtype=t2.dtype)
        else:
            score = score.to(dev)
            if score.dim() == 2:
                score = score.unsqueeze(-1)              # [B,L,1]
            elif score.dim() == 3 and score.size(-1) not in (1, D):
                score = score.mean(dim=-1, keepdim=True) # [B,L,1]

        # --- masks to device & safe shapes ---
        key_padding_mask = (key_padding_mask.to(dev).bool()
                            if key_padding_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=dev))
        if key_padding_mask.size(1) != L:
            # pad or crop to match L
            kpm = torch.zeros(B, L, dtype=torch.bool, device=dev)
            m = min(L, key_padding_mask.size(1))
            kpm[:, :m] = key_padding_mask[:, :m]
            key_padding_mask = kpm

        # atomic congruity
        q1 = torch.bmm(t2, v2.permute(0, 2, 1)) / math.sqrt(float(D))
        c = torch.sum(score * t2, dim=1, keepdim=True)  # [B,1,D]

        # token importance
        pa_token = self.linear1(t2).squeeze(-1)  # [B,L]
        pa_token = pa_token.masked_fill(key_padding_mask, float("-inf"))

        # NEW: safe softmax
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
            ], dim=0))  # [B,L,D]

        # image GAT
        v3 = v2
        # Ensure we pass a 2D edge_index per sample
        if isinstance(img_edge_index, (list, tuple)):
            iei_seq = [ei.to(dev) for ei in img_edge_index]
        elif isinstance(img_edge_index, torch.Tensor) and img_edge_index.dim() == 3:
            # Batched edge_index of shape [B, 2, E] -> list of [2, E]
            iei_seq = [img_edge_index[b].to(dev) for b in range(img_edge_index.size(0))]
        else:
            # Single graph: broadcast to all samples in the batch
            iei_seq = [img_edge_index.to(dev) for _ in range(v3.size(0))]

        for gat in self.img_conv:
            v3 = self.norm(torch.stack([
                self.relu1(gat(x, ei)) for x, ei in zip(v3, iei_seq)
            ], dim=0))  # [B,K,D]

        # compositional congruity
        tnp = torch.cat([tnp, c], dim=1)  # [B,L+1,D]
        q2 = torch.bmm(tnp, v3.permute(0, 2, 1)) / math.sqrt(float(tnp.size(2)))  # [B,L+1,K]

        # NP importance
        pa_np = self.linear2(tnp).squeeze(-1)  # [B, L+1]
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

        pa_np = probs.unsqueeze(2).expand(B, tnp.size(1), v3.size(1))  # [B, L+1, K]

        a_1 = torch.sum(q1 * pa_token, dim=1)  # [B, K]
        a_2 = torch.sum(q2 * pa_np,   dim=1)   # [B, K]
        return torch.cat([a_1, a_2], dim=1)    # [B, 2K]


