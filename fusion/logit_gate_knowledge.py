# fusion/logit_gate_knowledge.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from model_enhanced import Alignment, pool_tokens_to_words_batch

class MultiLogitGateFusion(nn.Module):
    def __init__(self, num_branches, num_classes, tau=2.0, eps=0.0, entropy_lambda=0.0):
        super().__init__()
        self.K = num_branches
        self.C = num_classes 
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps
        self.entropy_lambda = entropy_lambda

        # simple, stable gate (no need to be fancy)
        self.norm = nn.LayerNorm(num_branches)
        self.gate = nn.Linear(num_branches, num_branches)

    def forward(self, logits_list):
        # logits_list: list of [B, C] -> [B, K, C]
        L = torch.stack(logits_list, dim=1)                  # [B, K, C]
        B, K, C = L.shape

        # gate input: per-branch confidence (detach to avoid shortcuts)
        g_inp = L.detach().mean(-1)                          # [B, K]
        g = self.gate(self.norm(g_inp))                      # [B, K]
        w = F.softmax(g / self.tau, dim=1)                   # [B, K]

        # entropy regularization (toward uniform = log K)
        if self.training and self.entropy_lambda > 0.0 and K > 1:
            p = (w + self.eps) / (1.0 + self.eps * K)        # ε-floor (optional)
            ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(1)  # [B]
            target = math.log(K)
            gate_entropy_loss = self.entropy_lambda * ((ent - target) ** 2).mean()
        else:
            gate_entropy_loss = L.new_zeros(())

        fused = (w.unsqueeze(-1) * L).sum(1)                 # [B, C]
        return fused, gate_entropy_loss


class KnowledgeOnlyLogitGateModel(nn.Module):
    """
    Knowledge-only classifier (Captions/ANP/Attr subsets) with:
      - EnhancedTextEncoder to produce token text + per-type knowledge embeddings
      - Alignment to compute [a1_k, a2_k] per knowledge type
      - Per-type heads -> logits_k
      - MultiLogitGateFusion over available types
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, knowledge_types=[2,3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1, use_attention_single=True):
        super().__init__()
        self.txt_out_size = txt_out_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.lam = lam

        self.use_attention_single = use_attention_single

        # Text + knowledge encoders
        self.knowledge_encoder = EnhancedTextEncoder(
            input_size=txt_input_dim,
            output_size=txt_out_size,
            knowledge_types=knowledge_types,
            max_knowledge_length=max_knowledge_length,
            dropout=cro_drop
        )
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=txt_out_size,
            num_heads=cro_heads,
            dropout=cro_drop,
            num_knowledge_types=len(knowledge_types)
        )

        # Alignment (text words ↔ per-type knowledge vectors)
        self.knowledge_alignment = Alignment(
            input_size=txt_out_size,
            txt_gat_layer=txt_gat_layer,
            txt_gat_drop=txt_gat_drop,
            txt_gat_head=txt_gat_head,
            lam=lam
        )

        # One tiny head per knowledge type: maps [a1_k, a2_k] -> [2]
        self.type_heads = nn.ModuleList([nn.Linear(2, 2) for _ in knowledge_types])

        # N-way gated fusion over type logits
        self.fusion = MultiLogitGateFusion(num_branches=len(knowledge_types), num_classes=2)

    def forward(self, texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask,
                np_mask, knowledge_inputs, knowledge_masks):

        # Encode text + knowledge
        t_tok, t_scores_tok, k_type_emb, k_type_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        K = k_type_emb.size(1)

        if K == 1:
            # Identify single knowledge stream (1=caption, 2=ANP, 3=ATTR)
            ktype = int(self.knowledge_types[0]) if hasattr(self, "knowledge_types") else -1

            if ktype == 1:
                # ---- CAPTION-ONLY: keep identity (no attention) ----
                fused_text_tok = t_tok
                fused_k_type   = k_type_emb
                entropy_loss   = torch.zeros((), device=t_tok.device)

            else:
                # ---- ANP/ATTR-ONLY: one-way cross-attention (text <- knowledge) ----
                attn  = self.knowledge_fusion.attention
                ln_t  = self.knowledge_fusion.ln_text
                projt = self.knowledge_fusion.out_proj_text

                B, Lk, Dk = k_type_emb.shape
                if Lk > 1:
                    # Rare: if tags came as a sequence, pool to [B,1,D]
                    km = None
                    if knowledge_masks and len(knowledge_masks) > 0 and knowledge_masks[0] is not None:
                        km = knowledge_masks[0]
                        if km.dim() == 3 and km.size(-1) == 1:
                            km = km.squeeze(-1)                  # [B, Lk]
                        km = km.bool()
                        if km.size(1) != Lk:
                            L = min(km.size(1), Lk)
                            # pad with True (=ignore) up to Lk
                            pad = torch.ones(B, Lk - L, dtype=torch.bool, device=km.device)
                            km = torch.cat([km[:, :L], pad], dim=1)
                        keep  = (~km).float().unsqueeze(-1)      # [B,Lk,1]
                        denom = keep.sum(1, keepdim=True).clamp_min(1.0)
                        k_type_emb = (k_type_emb * keep).sum(1, keepdim=True) / denom  # [B,1,D]
                    else:
                        k_type_emb = k_type_emb.mean(dim=1, keepdim=True)             # [B,1,D]

                att_text, _ = attn(
                    query=t_tok,           # [B, Lt, D]
                    key=k_type_emb,        # [B, 1, D]
                    value=k_type_emb,
                    key_padding_mask=None  # single vector → no mask
                )

                fused_text_tok = ln_t(t_tok + projt(torch.nan_to_num(att_text)))
                fused_k_type   = k_type_emb
                entropy_loss   = torch.zeros((), device=t_tok.device)

        else:
            # ---- MULTI-KNOWLEDGE: learned attention/gating ----
            fused_text_tok, fused_k_type, _, entropy_loss = self.knowledge_fusion(
                text_embeddings=t_tok,
                knowledge_embeddings=k_type_emb,
                knowledge_masks=knowledge_masks,
                knowledge_scores=k_type_scores
            )

        # --- Tokens -> words for alignment (unchanged) ---
        fused_text_word, fused_text_word_scores = pool_tokens_to_words_batch(
            seq=fused_text_tok,
            score=t_scores_tok,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )

        B, K_eff, D = fused_k_type.shape
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_k_type.device)

        align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )  # [B, 2*K_eff]

        # Per-type tiny heads -> logits_k
        logits_list = []
        for i in range(K_eff):
            a1_k = align[:, i]
            a2_k = align[:, i + K_eff]
            a_pair = torch.stack([a1_k, a2_k], dim=-1)  # [B,2]
            logits_k = self.type_heads[i](a_pair)       # [B,2]
            logits_list.append(logits_k)

        if K_eff == 1:
            return logits_list[0], entropy_loss

        fused_logits, gate_ent = self.fusion(logits_list)
        total_ent = entropy_loss + gate_ent
        return fused_logits, total_ent

