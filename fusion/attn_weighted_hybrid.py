# fusion/attn_weighted_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from interraction.inter_models import CroModality
from model_enhanced import Alignment  # reuse existing

class AttnWeightedFusion(nn.Module):
    """
    Attention-weighted fusion over branch logits.
    Produces scalar weights per branch that sum to 1:
      w = softmax( W2 * tanh(W1 [logits_a || logits_b]) )  -> [B, 2]
      logits = w_a * logits_a + w_b * logits_b
    """
    def __init__(self, num_classes: int = 2, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(2 * num_classes, hidden)
        self.fc2 = nn.Linear(hidden, 2)  # 2 branches: text-image and knowledge

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([logits_a, logits_b], dim=-1)     # [B, 2C]
        h = torch.tanh(self.fc1(x))                     # [B, H]
        w = F.softmax(self.fc2(h), dim=-1)              # [B, 2]
        wa, wb = w[:, :1], w[:, 1:]                    # [B,1], [B,1]
        return wa * logits_a + wb * logits_b           # [B, C]


class HybridAttnWeightedModel(nn.Module):
    """
    Hybrid variant identical to your HybridModel pipeline,
    except the final fusion uses AttnWeightedFusion over the two branch logits.
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

        # interaction
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
        )

        # knowledge fusion + aligners
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

        # branch heads that output logits
        self.text_image_output = nn.Linear(2 * self.img_patch, 2)
        self.knowledge_head = nn.Linear(2, 2)

        # attention-weighted fusion
        self.fusion = AttnWeightedFusion(num_classes=2, hidden=16)

    @staticmethod
    def _pool_tokens_to_words(seq, score, word_spans, pad_len):
        from model_enhanced import pool_tokens_to_words_batch
        return pool_tokens_to_words_batch(seq, score, word_spans, pad_len)

    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,
                txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):

        # image branch
        v2, pv = self.img_encoder(imgs, lam=self.lam)

        # text + knowledge tokens
        t_tok, t_scores_tok, k_type_emb, _k_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        # cross-modal interaction
        v2, t_tok = self.interaction(images=v2, texts=t_tok, key_padding_mask=mask_batch)

        # tokens -> words for text-image alignment
        t_word, t_scores_word = self._pool_tokens_to_words(
            seq=t_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        # knowledge attention
        fused_text_tok, fused_k_type, _ = self.knowledge_fusion(
            text_embeddings=t_tok,
            knowledge_embeddings=k_type_emb,
            knowledge_masks=knowledge_masks
        )

        # tokens -> words for knowledge alignment
        fused_text_word, fused_text_word_scores = self._pool_tokens_to_words(
            seq=fused_text_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        B = fused_k_type.size(0)
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_k_type.device)

        # alignments
        ti_align = self.text_image_alignment(
            t2=t_word, v2=v2,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=t_scores_word, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=img_edge_index, lam=self.lam
        )

        know_align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )

        # heads → logits
        pv = torch.softmax(pv, dim=1).repeat(1, 2)           # [B, 2Kimg]
        logits_ti = self.text_image_output(ti_align * pv)     # [B,2]

        B_, twoK = know_align.shape
        if twoK == 0:
            pooled_know = torch.zeros(B_, 2, device=know_align.device, dtype=know_align.dtype)
        else:
            K = twoK // 2
            a1 = know_align[:, :K]
            a2 = know_align[:, K:]
            pooled_know = F.adaptive_avg_pool1d(torch.stack([a1, a2], dim=1), 1).squeeze(-1)  # [B,2]
        logits_k = self.knowledge_head(pooled_know)           # [B,2]

        # attention-weighted fusion over branch logits
        return self.fusion(logits_ti, logits_k)
