# fusion/early_film_hybrid.py
import torch
import torch.nn as nn
from images.image_models import ImageEncoder
from text.multi_knowledge_models import EnhancedTextEncoder
from interraction.inter_models import CroModality
from model_enhanced import Alignment, pool_tokens_to_words_batch


class FiLMLayer(nn.Module):
    """
    Residual Feature-wise Linear Modulation:
      y = (1 + gamma) * x + beta
    """
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(condition_dim, feature_dim)
        self.beta_net  = nn.Linear(condition_dim, feature_dim)
        # Start as identity
        nn.init.zeros_(self.gamma_net.weight); nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight);  nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], condition: [B, C]
        gamma = self.gamma_net(condition).unsqueeze(1)  # [B,1,D]
        beta  = self.beta_net(condition).unsqueeze(1)   # [B,1,D]
        return x * (1.0 + gamma) + beta


class FiLMEarlyFusion(nn.Module):
    """
    EARLY FUSION via FiLM (knowledge → image modulation BEFORE cross-modal interaction).
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
        self.txt_out_size  = txt_out_size
        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim   = img_out_dim if img_out_dim == txt_out_size else txt_out_size
        self.knowledge_types = list(knowledge_types)
        self.max_knowledge_length = max_knowledge_length
        self.cro_layers = cro_layers
        self.cro_heads  = cro_heads
        self.cro_drop   = cro_drop
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop  = txt_gat_drop
        self.txt_gat_head  = txt_gat_head
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop  = img_gat_drop
        self.img_gat_head  = img_gat_head
        self.img_patch = img_patch
        self.lam = lam
        self.type_bmco = type_bmco

        # Encoders
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

        # Knowledge → conditioning projection
        self.knowledge_pooler = nn.Sequential(
            nn.LayerNorm(self.txt_out_size),
            nn.Linear(self.txt_out_size, 256),
            nn.ReLU(),
            nn.Dropout(self.cro_drop),
        )

        # FiLM modulation on image features (EARLY)
        self.film = FiLMLayer(feature_dim=self.img_out_dim, condition_dim=256)

        # Cross-modal interaction (text ↔ images)
        self.interaction = CroModality(
            input_size=self.img_out_dim,
            nhead=self.cro_heads,
            dim_feedforward=2 * self.img_out_dim,
            dropout=self.cro_drop,
            cro_layer=self.cro_layers,
            type_bmco=self.type_bmco
        )

        # Alignment & head
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
        self.output_layer = nn.Sequential(
            nn.LayerNorm(2 * self.img_patch),
            nn.Linear(2 * self.img_patch, 2)
        )

    @staticmethod
    def _pool_knowledge_to_vec(k_type_emb: torch.Tensor) -> torch.Tensor | None:
        """
        Robustly pool per-type knowledge embeddings to [B, D]:
          - [B, K, L, D] -> mean over L then K
          - [B, K, D]    -> mean over K
          - [B, D]       -> as-is
          - None         -> None
        """
        if k_type_emb is None:
            return None
        if k_type_emb.dim() == 4:
            # [B, K, L, D] -> [B, K, D] -> [B, D]
            return k_type_emb.mean(dim=2).mean(dim=1)
        if k_type_emb.dim() == 3:
            # [B, K, D] -> [B, D]
            return k_type_emb.mean(dim=1)
        if k_type_emb.dim() == 2:
            # [B, D]
            return k_type_emb
        # Unexpected shape: be conservative
        return None

    def forward(self, imgs, texts, mask_batch, img_edge_index,
                t1_word_seq, txt_edge_index, gnn_mask, np_mask,
                knowledge_inputs, knowledge_masks):

        # 1) Encode text (+knowledge)
        t_tok, t_scores_tok, k_type_emb, _k_type_scores = self.txt_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )

        # Build conditioning vector from knowledge embeddings
        k_vec = self._pool_knowledge_to_vec(k_type_emb)  # [B, D] or None
        if k_vec is not None:
            condition = self.knowledge_pooler(k_vec)  # [B, 256]
        else:
            condition = torch.zeros(imgs.size(0), 256, device=imgs.device, dtype=imgs.dtype)

        # 2) Encode image and apply FiLM (EARLY)
        v2, pv = self.img_encoder(imgs, lam=self.lam)  # v2: [B, P, D], pv: [B, P]
        v2 = self.film(v2, condition)                  # FiLM-modulated image features

        # Move edge indices to device (safety)
        device = v2.device
        if isinstance(txt_edge_index, torch.Tensor): txt_edge_index = txt_edge_index.to(device)
        if isinstance(img_edge_index, torch.Tensor): img_edge_index = img_edge_index.to(device)

        # 3) Cross-modal interaction
        v2, t_tok = self.interaction(images=v2, texts=t_tok, key_padding_mask=mask_batch)

        # 4) Alignment & classification
        t_word, t_scores_word = pool_tokens_to_words_batch(
            seq=t_tok, score=t_scores_tok, word_spans=t1_word_seq, pad_len=mask_batch.size(1)
        )

        align_vec = self.alignment(
            t2=t_word, v2=v2,
            edge_index=txt_edge_index, gnn_mask=gnn_mask,
            score=t_scores_word, key_padding_mask=mask_batch, np_mask=np_mask,
            img_edge_index=img_edge_index, lam=self.lam
        )  # [B, 2*P]

        # Optional: weight by patch-importance prior from the image encoder
        pv_scaled = torch.softmax(pv, dim=1).repeat(1, 2)  # [B, 2*P]
        logits = self.output_layer(align_vec * pv_scaled)  # [B, 2]
        return logits
