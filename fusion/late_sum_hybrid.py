# fusion/late_sum_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from interraction.inter_models import CroModality
from model_enhanced import Alignment  # reuse your existing Alignment

class LateSumFusion(nn.Module):
    """
    Simple late fusion by summing class logits: logits = logits_a + logits_b
    """
    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        # Each is [B, 2]; return [B, 2]
        return logits_a + logits_b


class HybridLateSumModel(nn.Module):
    """
    Hybrid model variant that is identical to your current HybridModel
    EXCEPT the final fusion is late-sum of the two 2-class heads.
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768,
                 img_inter_dim=500, img_out_dim=300, knowledge_types=[1, 2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, img_gat_layer=2,
                 img_gat_drop=0.5, img_gat_head=2, img_patch=49, lam=1, type_bmco=1):
        super().__init__()

        # store
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

        # match dims
        if self.img_out_dim != self.txt_out_size:
            self.img_out_dim = self.txt_out_size

        # encoders
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            dropout=self.cro_drop
        )
        self.img_encoder = ImageEncoder(
            input_dim=self.img_input_dim,
            inter_dim=self.img_inter_dim,
            output_dim=self.img_out_dim
        )

        # interaction (text↔image)
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
        )

        # knowledge fusion & aligners
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=self.txt_out_size,
            num_heads=self.cro_heads,
            dropout=self.cro_drop
        )

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

        # heads (unchanged from your Hybrid: produce logits)
        self.text_image_output = nn.Linear(2 * self.img_patch, 2)
        self.knowledge_head = nn.Linear(2, 2)  # will take pooled [B,2] into logits

        # late-sum fusion
        self.late_sum = LateSumFusion()

    @staticmethod
    def _pool_tokens_to_words(seq, score, word_spans, pad_len):
        # inline import to avoid circulars
        from model_enhanced import pool_tokens_to_words_batch
        return pool_tokens_to_words_batch(seq, score, word_spans, pad_len)

    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,
                txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):

        # Encode image
        v2, pv = self.img_encoder(imgs, lam=self.lam)

        # Encode text + knowledge (token level)
        t_tok, t_scores_tok, k_type_emb, _k_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        # Cross-modal interaction on token level
        v2, t_tok = self.interaction(images=v2, texts=t_tok, key_padding_mask=mask_batch)

        # Pool tokens→words for text-image alignment
        t_word, t_scores_word = self._pool_tokens_to_words(
            seq=t_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        # Knowledge cross-attn (token text attends to per-type knowledge)
        fused_text_tok, fused_k_type, _ = self.knowledge_fusion(
            text_embeddings=t_tok,
            knowledge_embeddings=k_type_emb,
            knowledge_masks=knowledge_masks
        )

        # Pool tokens→words for knowledge alignment
        fused_text_word, fused_text_word_scores = self._pool_tokens_to_words(
            seq=fused_text_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        B = fused_k_type.size(0)
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_k_type.device)

        # Text–image alignment -> [B, 2Kimg]
        ti_align = self.text_image_alignment(
            t2=t_word, v2=v2,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=t_scores_word, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=img_edge_index, lam=self.lam
        )

        # Knowledge alignment -> [B, 2Kknow]
        know_align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )

        # Patch-importance weighting for text–image map (same as your Hybrid)
        pv = torch.softmax(pv, dim=1).repeat(1, 2)  # [B, 2Kimg]
        ti_aligned_weighted = ti_align * pv
        logits_ti = self.text_image_output(ti_aligned_weighted)  # [B,2]

        # Pool knowledge maps to [B,2], then map to logits
        B_, twoK = know_align.shape
        if twoK == 0:
            pooled_know_2 = torch.zeros(B_, 2, device=know_align.device, dtype=know_align.dtype)
        else:
            K = twoK // 2
            a1 = know_align[:, :K]   # [B,K]
            a2 = know_align[:, K:]   # [B,K]
            stacked = torch.stack([a1, a2], dim=1)              # [B,2,K]
            pooled_know_2 = F.adaptive_avg_pool1d(stacked, 1).squeeze(-1)  # [B,2]

        logits_k = self.knowledge_head(pooled_know_2)  # [B,2]

        # Late-sum fusion of logits
        logits = self.late_sum(logits_ti, logits_k)  # [B,2]
        return logits
