# fusion/logit_gate_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder
from interraction.inter_models import CroModality
from model_enhanced import Alignment, pool_tokens_to_words_batch
from fusion.logit_gate_knowledge import KnowledgeOnlyLogitGateModel


class LogitGateFusion(nn.Module):
    """
    Per-class gated late fusion on logits.
    alpha = sigmoid( W [logits_a || logits_b] )
    logits = alpha * logits_a + (1 - alpha) * logits_b
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * num_classes, num_classes),
            nn.Sigmoid()
        )

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        # logits_a, logits_b: [B, C]
        x = torch.cat([logits_a, logits_b], dim=-1)  # [B, 2C]
        alpha = self.gate(x)                         # [B, C]
        return alpha * logits_a + (1.0 - alpha) * logits_b


class HybridLogitGateModel(nn.Module):
    """
    Full hybrid:
      - Branch A: Text + Image → logits_ti
      - Branch B: Knowledge-only (Captions/ANP/Attr) → logits_k  (returns (logits, aux_loss))
      - Final: Logit-gated fusion between logits_ti and logits_k
    """
    def __init__(self,
                 txt_input_dim=768, txt_out_size=300,
                 img_input_dim=768, img_inter_dim=500, img_out_dim=300,
                 knowledge_types=(1, 2, 3),  # [CAP, ANP, ATTR]
                 max_knowledge_length=20,
                 cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2,
                 img_gat_layer=2, img_gat_drop=0.5, img_gat_head=2,
                 img_patch=49, lam=1, type_bmco=1):
        super().__init__()

        # store
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size
        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim if img_out_dim == txt_out_size else txt_out_size
        self.knowledge_types = list(knowledge_types)
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

        # --- Branch A: text + image encoders & interaction (no external knowledge here) ---
        self.txt_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=[],  # no external knowledge in this branch
            max_knowledge_length=0,
            dropout=self.cro_drop
        )
        self.img_encoder = ImageEncoder(
            input_dim=self.img_input_dim,
            inter_dim=self.img_inter_dim,
            output_dim=self.img_out_dim
        )
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
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
        self.text_image_output = nn.Linear(2 * self.img_patch, 2)

        # --- Branch B: knowledge-only (returns (logits, aux_loss)) ---
        self.knowledge_branch = KnowledgeOnlyLogitGateModel(
            txt_input_dim=self.txt_input_dim,
            txt_out_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            cro_layers=self.cro_layers,
            cro_heads=self.cro_heads,
            cro_drop=self.cro_drop,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            lam=self.lam
        )

        # --- Final modality gate ---
        self.fusion = LogitGateFusion(num_classes=2)

        # Optional: per-branch temperature
        self.register_buffer("temp_ti", torch.tensor(1.0))
        self.register_buffer("temp_k", torch.tensor(1.0))

        # place to stash aux loss for the trainer
        self._extra_loss = None

    def _pool_tokens_to_words(self, seq, score, word_spans, pad_len):
        return pool_tokens_to_words_batch(seq=seq, score=score, word_spans=word_spans, pad_len=pad_len)

    def forward(self, imgs, texts, mask_batch, img_edge_index,
                t1_word_seq, txt_edge_index, gnn_mask, np_mask,
                knowledge_inputs=None, knowledge_masks=None):

        device = imgs.device

        # Ensure graph edges are on the same device
        if isinstance(txt_edge_index, torch.Tensor):
            txt_edge_index = txt_edge_index.to(device)
        if isinstance(img_edge_index, torch.Tensor):
            img_edge_index = img_edge_index.to(device)

        # ===== Branch A: Text + Image → logits_ti =====
        v2, pv = self.img_encoder(imgs, lam=self.lam)  # v2: [B, P, D], pv: [B, P]

        t_tok, t_scores_tok, _k_type_emb_unused, _k_scores_unused = self.txt_encoder(
            text_input=texts, knowledge_inputs=None, knowledge_masks=None, lam=self.lam
        )

        v2, t_tok = self.interaction(images=v2, texts=t_tok, key_padding_mask=mask_batch)

        t_word, t_scores_word = self._pool_tokens_to_words(
            seq=t_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        ti_align = self.text_image_alignment(
            t2=t_word, v2=v2,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=t_scores_word, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=img_edge_index, lam=self.lam
        )  # [B, 2*img_patch]

        pv_scaled = torch.softmax(pv, dim=1).repeat(1, 2)  # [B, 2*Kimg]
        logits_ti = self.text_image_output(ti_align * pv_scaled)  # [B, 2]

        # ===== Branch B: Knowledge-only → (logits_k, aux_loss) or logits_k =====
        kb_out = self.knowledge_branch(
            texts=texts, mask_batch=mask_batch,
            t1_word_seq=t1_word_seq, txt_edge_index=txt_edge_index,
            gnn_mask=gnn_mask, np_mask=np_mask,
            knowledge_inputs=knowledge_inputs, knowledge_masks=knowledge_masks
        )

        if isinstance(kb_out, tuple):
            logits_k, extra = kb_out
            if isinstance(extra, torch.Tensor):
                self._extra_loss = (self._extra_loss or 0.0) + extra
        else:
            logits_k = kb_out

        # ===== Final gated fusion between branches =====
        if self.temp_ti.item() != 1.0:
            logits_ti = logits_ti / self.temp_ti
        if self.temp_k.item() != 1.0:
            logits_k = logits_k / self.temp_k

        return self.fusion(logits_ti, logits_k)
