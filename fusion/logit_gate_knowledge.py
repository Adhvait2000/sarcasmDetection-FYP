# fusion/logit_gate_knowledge.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from model_enhanced import Alignment, pool_tokens_to_words_batch

class MultiLogitGateFusion(nn.Module):
    def __init__(self, num_branches, num_classes, tau=2.0, eps=0.0, entropy_lambda=0.01):
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

         # ---- Uniform init: start near-equal weights across branches ----
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.zero_()

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
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1,
                 use_wka=False, gate_tau=2.0, gate_entropy_lambda=0.01):
        super().__init__()
        
        # Store all config (needed for save/load)
        self.txt_out_size = txt_out_size
        self.knowledge_types = knowledge_types
        self.max_knowledge_length = max_knowledge_length
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.lam = lam
        self.use_wka = use_wka
        
        # Encoder
        self.knowledge_encoder = EnhancedTextEncoder(
            input_size=txt_input_dim,
            output_size=txt_out_size,
            knowledge_types=knowledge_types,
            max_knowledge_length=max_knowledge_length,
            dropout=cro_drop
        )
        
        # Conditional fusion: WKA vs skip
        if use_wka:
            self.knowledge_fusion = WeightedKnowledgeAttention(
                input_size=txt_out_size,
                num_heads=cro_heads,
                dropout=cro_drop,
                num_knowledge_types=len(knowledge_types)
            )
        else:
            self.knowledge_fusion = None
        
        # Alignment
        self.knowledge_alignment = Alignment(
            input_size=txt_out_size,
            txt_gat_layer=txt_gat_layer,
            txt_gat_drop=txt_gat_drop,
            txt_gat_head=txt_gat_head,
            lam=lam
        )
        
        # Per-type tiny heads with dropout
        self.pre_head_dropout = nn.Dropout(0.15)
        self.type_heads = nn.ModuleList([
            nn.Linear(2, 2) for _ in knowledge_types
        ])
        
        # Late logit gate (only one initialization!)
        self.fusion = MultiLogitGateFusion(
            num_branches=len(knowledge_types),
            num_classes=2,
            tau=gate_tau,
            entropy_lambda=gate_entropy_lambda
        )

    def forward(self, texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask,
            np_mask, knowledge_inputs, knowledge_masks):
    
        # Encode text + per-type knowledge
        t_tok, t_scores_tok, k_type_emb, k_type_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # Ensure k_type_emb is [B, K, D]
        if k_type_emb.dim() == 4:  # [B,K,Lk,D] → pool over Lk
            k_type_emb = k_type_emb.mean(dim=2)
        
        K = k_type_emb.size(1)
        
        # --- Single-knowledge path ---
        if K == 1:
            fused_text_tok = t_tok
            fused_k_type = k_type_emb  # Already [B,1,D]
            entropy_loss = torch.zeros((), device=t_tok.device)
        else:
            # ---- Toggle WKA ----
            if self.use_wka and self.knowledge_fusion is not None:
                # Knowledge-type dropout (independent per type)
                if self.training:
                    B = k_type_emb.size(0)
                    drop_mask = torch.bernoulli(
                        torch.full((B, K, 1), 0.85, device=k_type_emb.device)
                    )
                    k_type_emb = k_type_emb * drop_mask
                
                fused_text_tok, fused_k_type, _, entropy_loss = self.knowledge_fusion(
                    text_embeddings=t_tok,
                    knowledge_embeddings=k_type_emb,
                    knowledge_masks=knowledge_masks,
                    knowledge_scores=k_type_scores
                )
            else:
                # Bypass WKA
                fused_text_tok = t_tok
                fused_k_type = k_type_emb
                entropy_loss = torch.zeros((), device=t_tok.device)
        
        # --- Tokens → words ---
        fused_text_word, fused_text_word_scores = pool_tokens_to_words_batch(
            seq=fused_text_tok,
            score=t_scores_tok,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )
        
        B, K_eff, D = fused_k_type.shape
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_k_type.device)
        
        # Alignment
        align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )  # [B, 2*K_eff]
        
        # Per-type heads with dropout
        logits_list = []
        for i in range(K_eff):
            a1_k = align[:, i]
            a2_k = align[:, i + K_eff]
            a_pair = torch.stack([a1_k, a2_k], dim=-1)  # [B,2]
            a_pair = self.pre_head_dropout(a_pair)
            logits_k = self.type_heads[i](a_pair)       # [B,2]
            logits_list.append(logits_k)
        
        if K_eff == 1:
            return logits_list[0], entropy_loss
        
        fused_logits, gate_ent = self.fusion(logits_list)
        total_ent = entropy_loss + gate_ent
        return fused_logits, total_ent

