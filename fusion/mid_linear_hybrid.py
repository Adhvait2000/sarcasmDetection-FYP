# fusion/mid_linear_hybrid.py
import torch
import torch.nn as nn

from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from interraction.inter_models import CroModality
from model_enhanced import Alignment, pool_tokens_to_words_batch


class HybridMidLinearModel(nn.Module):
    """
    MID-LEVEL LINEAR FUSION:
      - Branch A (Text↔Image): alignment map (pre-logits) -> proj
      - Branch B (Text↔Knowledge): alignment scores + semantic pool (pre-logits) -> proj
      - Fuse: concat([A_proj, B_proj]) -> classifier
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768,
                 img_inter_dim=500, img_out_dim=300, knowledge_types=(1, 2, 3),
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
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

        # knowledge fusion (token-text attends to per-type knowledge)
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=self.txt_out_size,
            num_heads=self.cro_heads,
            dropout=self.cro_drop
        )

        # alignment modules (feature maps; NO per-branch logits)
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

        # mid-level projections (map to common size)
        self.proj_ti = nn.Sequential(
            nn.LayerNorm(2 * self.img_patch),
            nn.Linear(2 * self.img_patch, self.txt_out_size),
            nn.ReLU(),
            nn.Dropout(self.cro_drop)
        )
        # knowledge: [2 (alignment pools) + txt_out_size (semantic)] -> txt_out_size
        self.proj_kn = nn.Sequential(
            nn.LayerNorm(self.txt_out_size + 2),
            nn.Linear(self.txt_out_size + 2, self.txt_out_size),
            nn.ReLU(),
            nn.Dropout(self.cro_drop)
        )

        # final classifier on concatenated reps
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.txt_out_size, self.txt_out_size),
            nn.ReLU(),
            nn.Dropout(self.cro_drop),
            nn.Linear(self.txt_out_size, 2)
        )

    # --- small helper: normalize per-type masks to [B,K] where 1=valid, 0=absent ---
    def _per_type_valid_mask(self, knowledge_masks, B, K, device):
        """
        Returns [B, K] float mask.
        Accepts:
          - None -> all valid
          - list of per-type masks: each [B, Lk] or [B, Lk, 1], True=pad
          - tensor [B, K] where True=pad/inactive (we invert)
        """
        if knowledge_masks is None:
            return torch.ones(B, K, device=device)

        if torch.is_tensor(knowledge_masks):
            m = knowledge_masks.to(device)
            if m.dim() == 2 and m.size(1) == K:
                return (~m.bool()).float()  # invert: True-pad -> 0 valid
            return torch.ones(B, K, device=device)

        if isinstance(knowledge_masks, list):
            cols = []
            for i in range(K):
                if i >= len(knowledge_masks) or knowledge_masks[i] is None:
                    cols.append(torch.zeros(B, 1, device=device))  # treat missing as invalid
                    continue
                mb = knowledge_masks[i].to(device)
                if mb.dim() == 3 and mb.size(-1) == 1:
                    mb = mb.squeeze(-1)
                # mb: True=pad; if ALL pad -> invalid, else valid
                all_pad = mb.bool().all(dim=1, keepdim=True)
                valid = (~all_pad).float()
                cols.append(valid)
            return torch.cat(cols, dim=1)

        return torch.ones(B, K, device=device)

    def forward(self, imgs, texts, mask_batch, img_edge_index,
                t1_word_seq, txt_edge_index, gnn_mask, np_mask,
                knowledge_inputs, knowledge_masks):

        # ----- Encode image -----
        v2, pv = self.img_encoder(imgs, lam=self.lam)  # v2: [B,P,D], pv: [B,P]
        device = v2.device

        # ensure edges on correct device
        if isinstance(txt_edge_index, torch.Tensor):
            txt_edge_index = txt_edge_index.to(device)
        if isinstance(img_edge_index, torch.Tensor):
            img_edge_index = img_edge_index.to(device)

        # ----- Encode text + knowledge (token level) -----
        t_tok, t_scores_tok, k_type_emb, k_type_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        # ----- Cross-modal interaction on token level -----
        v2, t_tok = self.interaction(images=v2, texts=t_tok, key_padding_mask=mask_batch)

        # tokens -> words for text-image alignment
        t_word, t_scores_word = pool_tokens_to_words_batch(
            seq=t_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        # ----- Knowledge cross-attn -----
        fused_text_tok, fused_k_type, _w, _ent = self.knowledge_fusion(
            text_embeddings=t_tok,
            knowledge_embeddings=k_type_emb,
            knowledge_masks=knowledge_masks,
            knowledge_scores=k_type_scores
        )

        # tokens -> words for knowledge alignment
        fused_text_word, fused_text_word_scores = pool_tokens_to_words_batch(
            seq=fused_text_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        B = fused_k_type.size(0)
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=device)

        # ----- ALIGNMENT FEATURES (representation level) -----
        # Text–image alignment: [B, 2*Kimg]
        ti_align = self.text_image_alignment(
            t2=t_word, v2=v2,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=t_scores_word, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=img_edge_index, lam=self.lam
        )

        # Knowledge alignment: [B, 2*Kknow]
        know_align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )

        # ----- MID-LEVEL REPRESENTATIONS -----
        # A) Text–Image: weight by patch-importance, then project
        pv_scaled = torch.softmax(pv, dim=1).repeat(1, 2)  # [B, 2*Kimg]
        ti_repr = self.proj_ti(ti_align * pv_scaled)       # [B, txt_out_size]

        # B) Knowledge: alignment pools (2) + semantic embedding (txt_out_size)
        B_, twoK = know_align.shape
        has_knowledge = (twoK > 0) and (fused_k_type.size(1) > 0) and (twoK % 2 == 0)

        if has_knowledge:
            K = twoK // 2
            # pooled alignment scores per class: [B,2]
            a1 = know_align[:, :K]   # [B,K]
            a2 = know_align[:, K:]   # [B,K]

            # per-type validity mask [B,K]
            valid = self._per_type_valid_mask(knowledge_masks, B=B_, K=K, device=device)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

            a1_mean = (a1 * valid).sum(dim=1, keepdim=True) / denom
            a2_mean = (a2 * valid).sum(dim=1, keepdim=True) / denom
            kn_pooled = torch.cat([a1_mean, a2_mean], dim=1)  # [B,2]

            # semantic pooling over available types -> [B, D]
            kn_semantic = (fused_k_type * valid.unsqueeze(-1)).sum(dim=1) / denom

            # concat scores+semantic -> project
            kn_emb = torch.cat([kn_pooled, kn_semantic], dim=-1)  # [B, 2 + D]
            kn_repr = self.proj_kn(kn_emb)                        # [B, D]
        else:
            kn_repr = torch.zeros(B_, self.txt_out_size, device=device, dtype=ti_repr.dtype)

        # ----- MID FUSION & CLASSIFY -----
        h = torch.cat([ti_repr, kn_repr], dim=-1)                 # [B, 2*D]
        logits = self.classifier(h)                                # [B, 2]
        return logits
