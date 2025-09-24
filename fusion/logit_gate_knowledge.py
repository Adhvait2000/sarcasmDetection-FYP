# fusion/logit_gate_knowledge.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from text.multi_knowledge_models import EnhancedTextEncoder, WeightedKnowledgeAttention
from model_enhanced import Alignment, pool_tokens_to_words_batch

class MultiLogitGateFusion(nn.Module):
    """
    N-branch, per-class soft mixture of logits.
    Inputs: list of logits [ [B,C], ..., K times ]
    Output: [B,C] with per-class gates over branches.
    """
    def __init__(self, num_branches: int, num_classes: int = 2):
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.gate = nn.Sequential(
            nn.Linear(num_branches * num_classes, num_branches * num_classes),
            nn.Sigmoid()
        )

    def forward(self, logits_list):
        # logits_list: length K, each [B,C]
        B, C = logits_list[0].shape
        K = len(logits_list)
        x = torch.cat(logits_list, dim=-1)                    # [B, K*C]
        g = self.gate(x).view(B, K, C)                        # [B,K,C]
        g = torch.softmax(g, dim=1)                           # per-class weights over branches
        stacked = torch.stack(logits_list, dim=1)             # [B,K,C]
        out = (g * stacked).sum(dim=1)                        # [B,C]
        return out

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
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1):
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

        # Encode tokens
        t_tok, t_scores_tok, k_type_emb, _k_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        # Cross attention to get (i) text contextualized by knowledge, (ii) per-type knowledge reprs
        fused_text_tok, fused_k_type, _ = self.knowledge_fusion(
            text_embeddings=t_tok,               # [B, Lt, D]
            knowledge_embeddings=k_type_emb,     # [B, K, D] where K=len(knowledge_types)
            knowledge_masks=knowledge_masks
        )

        # Tokens -> words for alignment
        fused_text_word, fused_text_word_scores = pool_tokens_to_words_batch(
            seq=fused_text_tok,
            score=t_scores_tok,
            word_spans=t1_word_seq,
            pad_len=mask_batch.size(1)
        )

        B, K, D = fused_k_type.shape
        # Dummy img_edge_index since this is knowledge-only
        dummy_img_ei = torch.zeros((B, 2, 0), dtype=torch.long, device=fused_k_type.device)

        # Alignment produces [B, 2K] = concat over k of (a1_k, a2_k blocks)
        align = self.knowledge_alignment(
            t2=fused_text_word, v2=fused_k_type,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=fused_text_word_scores, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=dummy_img_ei, lam=self.lam
        )  # [B, 2K]

        # Split per-type, make per-type logits, fuse with gates
        logits_list = []
        for i in range(K):
            a1_k = align[:, i]          # [B]
            a2_k = align[:, i + K]      # [B]
            a_pair = torch.stack([a1_k, a2_k], dim=-1)  # [B,2]
            logits_k = self.type_heads[i](a_pair)       # [B,2]
            logits_list.append(logits_k)

        if len(logits_list) == 1:
            return logits_list[0]  # single-type ablation degenerates to its own logits

        return self.fusion(logits_list)  # gated sum across the selected knowledge types
