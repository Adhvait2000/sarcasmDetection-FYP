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

class KnowledgeOnlyModel(nn.Module):
    """
    Knowledge-Only Model: ANPs + Attributes (no image patches)
    Uses only extracted knowledge without image features
    """
    def __init__(self, txt_input_dim=768, txt_out_size=300, knowledge_types=[2, 3],
                 max_knowledge_length=20, cro_layers=6, cro_heads=5, cro_drop=0.5,
                 txt_gat_layer=2, txt_gat_drop=0.5, txt_gat_head=2, lam=1):
        super(KnowledgeOnlyModel, self).__init__()
        
        # Model parameters
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
        
        # Enhanced text encoder for knowledge only
        self.knowledge_encoder = EnhancedTextEncoder(
            input_size=self.txt_input_dim,
            output_size=self.txt_out_size,
            knowledge_types=self.knowledge_types,
            max_knowledge_length=self.max_knowledge_length,
            dropout=self.cro_drop
        )
        
        # Knowledge fusion
        self.knowledge_fusion = WeightedKnowledgeAttention(
            input_size=self.txt_out_size,
            num_heads=self.cro_heads,
            dropout=self.cro_drop
        )
        
        # Knowledge-only alignment
        self.knowledge_alignment = Alignment(
            input_size=self.txt_out_size,
            txt_gat_layer=self.txt_gat_layer,
            txt_gat_drop=self.txt_gat_drop,
            txt_gat_head=self.txt_gat_head,
            lam=self.lam
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.txt_out_size, 2)
        
    def forward(self, texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, 
                np_mask, knowledge_inputs, knowledge_masks):
        """
        Forward pass for knowledge-only model
        
        Args:
            texts: Text input dictionary
            mask_batch: Text padding masks
            t1_word_seq: Word sequences
            txt_edge_index: Text edge indices
            gnn_mask: GNN masks
            np_mask: Noun phrase masks
            knowledge_inputs: Knowledge inputs (ANPs + attributes)
            knowledge_masks: Knowledge masks
            
        Returns:
            predictions: Model predictions
        """
        # Encode text and knowledge
        texts, text_scores, knowledge_embeddings, knowledge_scores = self.knowledge_encoder(
            text_input=texts,
            knowledge_inputs=knowledge_inputs,
            knowledge_masks=knowledge_masks,
            lam=self.lam
        )
        
        # Knowledge fusion
        fused_text, fused_knowledge, attention_weights = self.knowledge_fusion(
            text_embeddings=texts,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_masks=knowledge_masks
        )
        
        # Knowledge-only alignment
        alignment_scores = self.knowledge_alignment(
            t2=fused_text,
            v2=fused_knowledge,
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=text_scores,
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            lam=self.lam
        )
        
        # Final prediction
        predictions = self.output_layer(alignment_scores)
        
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
        
    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq,
                txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks):
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
        
        # Knowledge fusion
        fused_text, fused_knowledge, attention_weights = self.knowledge_fusion(
            text_embeddings=texts,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_masks=knowledge_masks
        )
        
        # Text-image alignment
        text_image_alignment = self.text_image_alignment(
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
        
        # Knowledge alignment
        knowledge_alignment = self.knowledge_alignment(
            t2=fused_text,
            v2=fused_knowledge,
            edge_index=txt_edge_index,
            gnn_mask=gnn_mask,
            score=knowledge_scores,
            key_padding_mask=mask_batch,
            np_mask=np_mask,
            lam=self.lam
        )
        
        # Apply importance weights
        pv = pv.repeat(1, 2)
        weighted_text_image = text_image_alignment * pv
        
        # Generate predictions
        text_image_pred = self.text_image_output(weighted_text_image)
        knowledge_pred = self.knowledge_output(knowledge_alignment)
        
        # Final fusion
        combined_features = torch.cat([text_image_pred, knowledge_pred], dim=-1)
        predictions = self.final_fusion(combined_features)
        
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
        
        # GAT layers
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
            ) for i in range(self.txt_gat_layer)
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
            ) for i in range(self.img_gat_layer)
        ])
        
        # Importance scoring
        self.linear1 = nn.Linear(self.input_size, 1)
        self.linear2 = nn.Linear(self.input_size, 1)
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.ReLU()
        
    def forward(self, t2, v2, edge_index, gnn_mask, score, key_padding_mask, 
                np_mask, img_edge_index, lam=1):
        """
        Forward pass for alignment computation
        """
        # Atomic level congruity
        q1 = torch.bmm(t2, v2.permute(0, 2, 1)) / math.sqrt(t2.size(2))
        c = torch.sum(score * t2, dim=1, keepdim=True)
        
        # Token importance
        pa_token = self.linear1(t2).squeeze().masked_fill_(key_padding_mask, float("-Inf"))
        
        # GAT processing
        tnp = t2
        for gat in self.txt_conv:
            tnp = self.norm(torch.stack([
                (self.relu1(gat(data[0], data[1].cuda(), mask=data[2])))
                for data in zip(tnp, edge_index, gnn_mask)
            ]))
        
        v3 = v2
        for gat in self.img_conv:
            v3 = self.norm(torch.stack([
                self.relu1(gat(data, img_edge_index.cuda()))
                for data in v3
            ]))
        
        # Compositional level congruity
        tnp = torch.cat([tnp, c], dim=1)
        q2 = torch.bmm(tnp, v3.permute(0, 2, 1)) / math.sqrt(tnp.size(2))
        
        # NP importance
        pa_np = self.linear2(tnp).squeeze().masked_fill_(np_mask, float("-Inf"))
        pa_np = F.softmax(pa_np * lam, dim=1).unsqueeze(2).repeat((1, 1, v3.size(1)))
        pa_token = F.softmax(pa_token * lam, dim=1).unsqueeze(2).repeat((1, 1, v3.size(1)))
        
        # Final alignment
        a_1 = torch.sum(q1 * pa_token, dim=1)
        a_2 = torch.sum(q2 * pa_np, dim=1)
        a = torch.cat([a_1, a_2], dim=1)
        
        return a