"""
Enhanced Training Script
Supports all three model variants with comprehensive evaluation
"""
import argparse
from tqdm import tqdm
import os
from utils.logging.tf_logger import SimpleLogger as Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
import time
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_enhanced import BaselineModel, KnowledgeOnlyModel, HybridModel, ImageOnlyModel
from utils.enhanced_dataset import EnhancedBaseSet, MultiKnowledgePadCollate
from utils.data_utils import construct_edge_image, seed_everything
from utils.compute_scores import get_metrics, get_four_metrics
from fusion.logit_gate_knowledge import KnowledgeOnlyLogitGateModel
from fusion.logit_gate_hybrid import HybridLogitGateModel

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.multiprocessing.set_sharing_strategy('file_system')

def _move_tokenizer_batch_to_device(batch_dict, device):
    if batch_dict is None:
        return None
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_dict.items()}

    
class EnhancedTrainer:
    def __init__(self, model_type="baseline", parameter_file="parameter.json"):
        """
        Enhanced trainer for different model variants
        
        Args:
            model_type: "baseline", "knowledge_only", or "hybrid"
            parameter_file: Path to parameter file
        """
        self.model_type = model_type
        
        # Load parameters
        with open(parameter_file) as f:
            self.parameter = json.load(f)
        
        # Initialize model based on type
        self.model = self._initialize_model()
        self.model.to(device=device)
        
        # Setup optimizer with proper learning rates
        head_lr = self.parameter.get("head_lr", self.parameter["lr"])

        bert_params, new_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            if "bert_model" in n or "bert" in n.lower():
                bert_params.append(p)
            else:
                new_params.append(p)
        
        param_groups = []
        if bert_params:
            param_groups.append({"params": bert_params, "lr": self.parameter["lr"]})
        if new_params:
            param_groups.append({"params": new_params, "lr": head_lr})
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": self.parameter["lr"]}]
        
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.parameter.get("weight_decay", 0.005),
            amsgrad=True
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=self.parameter.get("patience", 3)
        )
        
        # self.criterion = CrossEntropyLoss()

        self.criterion = self._make_criterion()
        
        # Initialize logger
        self.logger = Logger(f"logs/{model_type}_training")

        # Early Stopping parameters
        self.early_stopping_patience = self.parameter.get("early_stopping_patience", 3)
        self.early_stop_min_delta = self.parameter.get("early_stop_min_delta", 0.001)

        self.KNOW_ABLATIONS = {
        "anp_attr",
        "knowledge_only_anp",
        "knowledge_only_attr",
        "caption_only",
        "caption_anp",
        "caption_attr",
        "knowledge_combined"
        }  

    def _predict_probs_labels(self, data_loader):
        """Collect p(class=1) and labels from a loader. Tuple-safe for model outputs."""
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Scoring"):
                # --- Forward (mirror your evaluate() branching) ---
                if self.model_type in self.KNOW_ABLATIONS:
                    texts, mask_batch, word_spans, txt_edge_index, gnn_mask, np_mask, \
                    knowledge_inputs, knowledge_masks = self._prepare_knowledge_only_batch(batch)
                    outputs = self.model(
                        texts=texts, mask_batch=mask_batch, t1_word_seq=word_spans,
                        txt_edge_index=txt_edge_index, gnn_mask=gnn_mask, np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs, knowledge_masks=knowledge_masks,
                    )
                elif self.model_type == "image_only":
                    imgs = batch[0].to(device)
                    outputs = self.model(imgs=imgs)
                else:
                    imgs, texts, mask_batch, img_edge_index, word_spans, txt_edge_index, \
                    gnn_mask, np_mask, knowledge_inputs, knowledge_masks = self._prepare_batch(batch)
                    outputs = self.model(
                        imgs=imgs, texts=texts, mask_batch=mask_batch, img_edge_index=img_edge_index,
                        t1_word_seq=word_spans, txt_edge_index=txt_edge_index, gnn_mask=gnn_mask,
                        np_mask=np_mask, knowledge_inputs=knowledge_inputs, knowledge_masks=knowledge_masks,
                    )

                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = F.softmax(logits, dim=1)[:, 1]  # p(class=1)
                all_probs.append(probs.detach().cpu())
                all_labels.append(batch[8].to(device).long().detach().cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return all_labels, all_probs


    def _find_best_threshold(self, y_true, y_prob, average="weighted", grid=None):
        """Grid-search threshold ∈ [0,1] to maximize F1 on val."""
        import numpy as np
        from sklearn.metrics import f1_score

        if grid is None:
            grid = np.linspace(0.05, 0.95, 181)  # 0.05 step

        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            if average == "binary":
                f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
            else:
                f1 = f1_score(y_true, y_pred, average=average)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t, best_f1


    def _metrics_from_probs(self, y_true, y_prob, threshold, average="weighted"):
        """Compute metrics at a fixed threshold."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
        cm = confusion_matrix(y_true, y_pred)
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "confusion_matrix": cm}


    def _make_criterion(self):
        """
        Creates the loss based on parameter.json config.

        Supported keys:
          - "loss": "ce" (default)
          - "class_weights": [w0, w1]  # explicit per-class weights
          - "pos_weight": float        # convenience for binary: expands to [1.0, pos_weight]
          - "label_smoothing": float   # PyTorch CE label smoothing
        """
        cfg = self.parameter
        loss_type = cfg.get("loss", "ce").lower()

        # Option A: explicit per-class weights
        class_weights = cfg.get("class_weights", None)

        # Option B: convenience positive-class upweight (overrides nothing if class_weights is provided)
        if class_weights is None and "pos_weight" in cfg:
            class_weights = [1.0, float(cfg["pos_weight"])]

        label_smoothing = float(cfg.get("label_smoothing", 0.0))

        if loss_type == "ce":
            weight_tensor = None
            if class_weights is not None:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            return CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

        # Default fallback to CE if an unknown loss is specified
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        
    def _initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == "baseline":
            return BaselineModel(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                img_gat_layer=self.parameter["img_gat_layer"],
                img_gat_drop=self.parameter["img_gat_drop"],
                img_gat_head=self.parameter["img_gat_head"],
                img_patch=self.parameter["img_patch"],
                lam=self.parameter["lambda"],
                type_bmco=self.parameter["type_bmco"]
            )
        elif self.model_type == "image_only":
            return ImageOnlyModel(
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                img_patch=self.parameter["img_patch"],
                drop=self.parameter.get("cro_drop", 0.5),
                lam=self.parameter["lambda"],
            )
        elif self.model_type == "text_image":
            return BaselineModel(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                img_gat_layer=self.parameter["img_gat_layer"],
                img_gat_drop=self.parameter["img_gat_drop"],
                img_gat_head=self.parameter["img_gat_head"],
                img_patch=self.parameter["img_patch"],
                lam=self.parameter["lambda"],
                type_bmco=self.parameter["type_bmco"]
            )
        elif self.model_type in ["anp_attr", "knowledge_only_anp", 
                           "knowledge_only_attr", "caption_anp", "caption_attr", "caption_only", "knowledge_combined"]:
            # All knowledge ablations use the logit gate model
            return KnowledgeOnlyLogitGateModel(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                knowledge_types=self._get_knowledge_types(),
                max_knowledge_length=self.parameter.get("know_max_length", 20),
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                lam=self.parameter["lambda"],
                gate_tau=self.parameter.get("gate_tau", 2.0),
                 gate_entropy_lambda=self.parameter.get("gate_entropy_lambda", 0.01),
            )
        elif self.model_type == "hybrid":
            return HybridModel(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                knowledge_types=[1, 2, 3],  # Captions, ANP, attributes
                max_knowledge_length=self.parameter.get("know_max_length", 20),
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                img_gat_layer=self.parameter["img_gat_layer"],
                img_gat_drop=self.parameter["img_gat_drop"],
                img_gat_head=self.parameter["img_gat_head"],
                img_patch=self.parameter["img_patch"],
                lam=self.parameter["lambda"],
                type_bmco=self.parameter["type_bmco"]
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_knowledge_types(self):
        if self.model_type == "baseline":
            return []  # tweets
        elif self.model_type in ["image_only", "text_image"]:
            return []
        elif self.model_type == "anp_attr":
            return [2, 3]          # ANP + ATTR
        elif self.model_type == "caption_only":
            return [1]          # CAP only
        elif self.model_type == "knowledge_only_anp":
            return [2]             # ANP only
        elif self.model_type == "knowledge_only_attr":
            return [3]             # ATTR only
        elif self.model_type == "caption_anp":
            return [1, 2]          # CAP + ANP
        elif self.model_type == "caption_attr":
            return [1, 3]          # CAP + ATTR
        elif self.model_type == "knowledge_combined":
            return [1, 2, 3]        # CAP + ANP + ATTR
        elif self.model_type == "hybrid":   
            return [1, 2, 3]        # CAP + ANP + ATTR


    def _create_data_loaders(self):
        """Create data loaders for the specific model type"""
        knowledge_types = self._get_knowledge_types()

        # Dataset percentage handling
        dataset_percentage = self.parameter.get("dataset_percentage", 100.0)
        if dataset_percentage > 1.0:
            dataset_percentage = dataset_percentage / 100.0
        
        print(f"Dataset sampling: {dataset_percentage*100:.1f}%")
        print(f"Knowledge types for {self.model_type}: {knowledge_types}")

        # File names 
        train_img_file = self.parameter.get("train_img_file", "train_B32.pt")
        val_img_file = self.parameter.get("val_img_file", "val_B32.pt")
        test_img_file = self.parameter.get("test_img_file", "test_B32.pt")

        cache_dir = self.parameter.get("cache_dir")
        anp_cache = os.path.join(cache_dir, "anp_attr_all.jsonl")
        cap_cache = os.path.join(cache_dir, "captions_all.jsonl")

        def _assert_file(p, label):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found at: {p}")
            return p

        need_anp = (2 in knowledge_types) or (3 in knowledge_types)
        need_cap = (1 in knowledge_types)
        
        anp_cache = _assert_file(anp_cache, "ANP/Attr cache") if need_anp else None
        cap_cache = _assert_file(cap_cache, "Captions cache") if need_cap else None


        # Datasets
        train_dataset = EnhancedBaseSet(
            type="train",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "train.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], train_img_file),
            knowledge_types=knowledge_types,
            anp_attr_cache_path=anp_cache,
            caption_cache_path=cap_cache,   
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=dataset_percentage,
        )

        val_dataset = EnhancedBaseSet(
            type="val",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "val.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], val_img_file),
            knowledge_types=knowledge_types,
            anp_attr_cache_path=anp_cache,
            caption_cache_path=cap_cache, 
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=1.0,
        )

        test_dataset = EnhancedBaseSet(
            type="test",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "test.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], test_img_file),
            knowledge_types=knowledge_types,
            anp_attr_cache_path=anp_cache,
            caption_cache_path=cap_cache, 
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=1.0,
        )

        # Collate
        collate_fn = MultiKnowledgePadCollate(
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
        )

        # DataLoaders
        num_workers = self.parameter.get("num_workers", 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader

    def _construct_image_edge_index(self, batch_size, num_patches=49):
            """
            Construct image edge indices for the batch using the existing construct_edge_image function.
            """
            # Use the existing construct_edge_image function from data_utils
            # This creates edges connecting each patch to its 8 neighbors in the grid
            single_edge_index = construct_edge_image(num_patches)  # Returns [2, num_edges]
            
            # Replicate for each sample in the batch
            batch_edge_indices = []
            for b in range(batch_size):
                batch_edge_indices.append(single_edge_index)
            
            # Stack into batch tensor [B, 2, E]
            # All samples have the same edge structure for images
            batch_tensor = torch.stack([single_edge_index for _ in range(batch_size)])
            
            return batch_tensor
        
    def train_epoch(self, train_loader):
        """Train for one epoch (tuple-safe across all model variants)."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_entropy_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Debug first batch
            if batch_idx == 0:
                try:
                    img_shape = batch[0].shape if hasattr(batch[0], "shape") else "None"
                except Exception:
                    img_shape = "N/A"
                print("\nFirst batch shapes:")
                print(f"  Images: {img_shape}")
                print(f"  Batch length: {len(batch)}")

            # ---- Forward pass (by model type) ----
            if self.model_type in self.KNOW_ABLATIONS:
                texts, mask_batch, word_spans, txt_edge_index, gnn_mask, np_mask, \
                knowledge_inputs, knowledge_masks = self._prepare_knowledge_only_batch(batch)

                outputs = self.model(
                    texts=texts,
                    mask_batch=mask_batch,
                    t1_word_seq=word_spans,
                    txt_edge_index=txt_edge_index,
                    gnn_mask=gnn_mask,
                    np_mask=np_mask,
                    knowledge_inputs=knowledge_inputs,
                    knowledge_masks=knowledge_masks,
                )

            elif self.model_type == "image_only":
                imgs = batch[0].to(device)
                outputs = self.model(imgs=imgs)

            else:
                # baseline, text_image, hybrid
                imgs, texts, mask_batch, img_edge_index, word_spans, txt_edge_index, \
                gnn_mask, np_mask, knowledge_inputs, knowledge_masks = self._prepare_batch(batch)

                if batch_idx == 0:
                    print(f"  Prepared imgs: {imgs.shape}")
                    print(f"  img_edge_index: {img_edge_index.shape}")

                outputs = self.model(
                    imgs=imgs,
                    texts=texts,
                    mask_batch=mask_batch,
                    img_edge_index=img_edge_index,
                    t1_word_seq=word_spans,
                    txt_edge_index=txt_edge_index,
                    gnn_mask=gnn_mask,
                    np_mask=np_mask,
                    knowledge_inputs=knowledge_inputs,
                    knowledge_masks=knowledge_masks,
                )

            # ---- Normalize outputs to (logits, entropy_loss) for ALL branches ----
            if isinstance(outputs, tuple):
                logits = outputs[0]
                # use 2nd element if it's a tensor, else 0.0
                entropy_loss = outputs[1] if (len(outputs) > 1 and torch.is_tensor(outputs[1])) \
                            else torch.tensor(0.0, device=device)
            else:
                logits = outputs
                entropy_loss = torch.tensor(0.0, device=device)

            # ---- Loss & backward ----
            labels = batch[8].to(device).long()  # CE expects Long targets

            ce_loss = self.criterion(logits, labels)
            loss = ce_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()

            max_grad_norm = self.parameter.get("grad_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            # ---- Stats ----
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_entropy_loss += entropy_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                "CE_Loss": f"{ce_loss.item():.4f}",
                "Ent_Loss": f"{entropy_loss.item():.4f}",
                "Acc": f"{100.0 * correct / max(total,1):.2f}%"
            })

        avg_loss = total_loss / max(len(train_loader), 1)
        avg_ce = total_ce_loss / max(len(train_loader), 1)
        avg_ent = total_entropy_loss / max(len(train_loader), 1)

        print(f"  [Train Summary] Total: {avg_loss:.4f} (CE: {avg_ce:.4f}, Entropy: {avg_ent:.4f})")

        return avg_loss, 100.0 * correct / max(total, 1)

    
    def evaluate(self, data_loader, split="val"):
        """Evaluate model performance (tuple-safe; ignores entropy in eval)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                if self.model_type in self.KNOW_ABLATIONS:
                    texts, mask_batch, word_spans, txt_edge_index, gnn_mask, np_mask, \
                    knowledge_inputs, knowledge_masks = self._prepare_knowledge_only_batch(batch)

                    outputs = self.model(
                        texts=texts,
                        mask_batch=mask_batch,
                        t1_word_seq=word_spans,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks,
                    )

                elif self.model_type == "image_only":
                    imgs = batch[0].to(device)
                    outputs = self.model(imgs=imgs)

                else:
                    imgs, texts, mask_batch, img_edge_index, word_spans, txt_edge_index, \
                    gnn_mask, np_mask, knowledge_inputs, knowledge_masks = self._prepare_batch(batch)

                    outputs = self.model(
                        imgs=imgs,
                        texts=texts,
                        mask_batch=mask_batch,
                        img_edge_index=img_edge_index,
                        t1_word_seq=word_spans,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks,
                    )

                # ---- Normalize outputs to logits (ignore entropy in eval) ----
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                labels = batch[8].to(device).long()
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )
        cm = confusion_matrix(all_labels, all_predictions)

        return {
            "loss": total_loss / max(len(data_loader), 1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }


    
    def _prepare_batch(self, batch):
        """
        FIXED: Properly prepare batch for baseline and hybrid models
        """
        imgs = batch[0].to(device)
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]
        mask_batch = batch[4].to(device)
        txt_edge_index = batch[5]  # This is text edge index
        gnn_mask = batch[6].to(device)
        np_mask = batch[7].to(device)
        
        # CRITICAL FIX: Construct proper image edge indices
        batch_size = imgs.size(0)
        img_edge_index = self._construct_image_edge_index(batch_size, self.parameter["img_patch"])
        img_edge_index = img_edge_index.to(device)
        
        texts = _move_tokenizer_batch_to_device(texts, device)

        # Handle knowledge data
        knowledge_inputs = None
        knowledge_masks = None
        
        if len(batch) > 9:
            knowledge_inputs = []
            knowledge_masks = []
            
            # Extract knowledge inputs and masks
            for i in range(9, len(batch), 3):
                if i < len(batch) and batch[i] is not None:
                    kd = _move_tokenizer_batch_to_device(batch[i], device)  # move dict to device
                    knowledge_inputs.append(kd)
                    if i+2 < len(batch) and batch[i+2] is not None:
                        knowledge_masks.append(batch[i+2].to(device))
                    else:
                        knowledge_masks.append(None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)
        
        return imgs, texts, mask_batch, img_edge_index, word_spans, txt_edge_index, \
               gnn_mask, np_mask, knowledge_inputs, knowledge_masks
    
    def _prepare_knowledge_only_batch(self, batch):
        """Prepare batch for knowledge-only model"""
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]
        mask_batch = batch[4].to(device)
        txt_edge_index = batch[5]
        gnn_mask = batch[6].to(device)
        np_mask = batch[7].to(device)

        texts = _move_tokenizer_batch_to_device(texts, device)
        
        # Handle knowledge data
        knowledge_inputs = []
        knowledge_masks = []
        
        if len(batch) > 9:
            for i in range(9, len(batch), 3):
                if i < len(batch) and batch[i] is not None:
                    kd = _move_tokenizer_batch_to_device(batch[i], device)
                    knowledge_inputs.append(kd)
                    if i+2 < len(batch) and batch[i+2] is not None:
                        knowledge_masks.append(batch[i+2].to(device))
                    else:
                        knowledge_masks.append(None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)
        
        return texts, mask_batch, word_spans, txt_edge_index, gnn_mask, np_mask, \
               knowledge_inputs, knowledge_masks
    
    def save_model(self, epoch, metrics, save_dir="saved_models"):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'model_type': self.model_type,
            'parameter': self.parameter
        }
        path = f"{save_dir}/{self.model_type}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        return path
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nTraining {self.model_type} model...")
        print(f"Device: {device}")
        print(f"Batch size: {self.parameter['batch_size']}")
        print(f"Learning rate: {self.parameter['lr']}")
        print(f"Epochs: {num_epochs}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()
        
        best_val_f1 = float('-inf')
        best_epoch = -1
        best_ckpt_path = None
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader, "val")
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.logger.log_scalar('train_loss', train_loss, epoch)
            self.logger.log_scalar('train_acc', train_acc, epoch)
            self.logger.log_scalar('val_loss', val_metrics['loss'], epoch)
            self.logger.log_scalar('val_acc', val_metrics['accuracy'], epoch)
            self.logger.log_scalar('val_f1', val_metrics['f1'], epoch)
            
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Check for improvement
            improved = (val_metrics['f1'] - best_val_f1) > self.early_stop_min_delta
            if improved:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                best_ckpt_path = self.save_model(epoch, val_metrics)
                print(f"\n[NEW BEST] Saved model with Val F1: {best_val_f1:.4f}")
                
                # Save confusion matrix
                self.plot_confusion_matrix(
                    val_metrics['confusion_matrix'],
                    f"confusion_matrix_{self.model_type}_best.png"
                )
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")
                
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model for testing
        if best_ckpt_path:
            checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {best_epoch+1}")
        else:
            print("\n[WARN] No best checkpoint recorded; proceeding with current weights.")

        print(f"\n{'='*50}")
        print(f"Tuning decision threshold on VAL (maximize F1)")
        print(f"{'='*50}")

        avg_for_f1 = self.parameter.get("threshold_f1_average", "weighted")  # "weighted" or "binary"

        val_y, val_p = self._predict_probs_labels(val_loader)
        best_t, best_val_f1_tuned = self._find_best_threshold(val_y, val_p, average=avg_for_f1)
        print(f"Best threshold on VAL: {best_t:.3f}  |  Tuned VAL F1 ({avg_for_f1}): {best_val_f1_tuned:.4f}")

        print(f"\n{'='*50}")
        print(f"Testing best model at tuned threshold t={best_t:.3f}")
        print(f"{'='*50}")

        test_y, test_p = self._predict_probs_labels(test_loader)
        test_metrics = self._metrics_from_probs(test_y, test_p, best_t, average=avg_for_f1)

        print(f"\nTest Results (thresholded at {best_t:.3f}):")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")

        # Optional: save the thresholded confusion matrix
        self.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            f"confusion_matrix_{self.model_type}_test_thresholded.png"
        )

        safe_best_epoch = best_epoch + 1 if best_epoch >= 0 else num_epochs
        results = {
            'model_type': self.model_type,
            'best_epoch': safe_best_epoch,
            'best_val_f1': best_val_f1,               # argmax path you tracked during training
            'val_best_threshold': float(best_t),
            'val_f1_tuned': float(best_val_f1_tuned),
            'threshold_average': avg_for_f1,
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1'])
            }
        }
        results_file = f"results_{self.model_type}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline','anp_attr','knowledge_combined','image_only','text_image',
         'knowledge_only_anp','knowledge_only_attr','caption_anp','caption_attr', 'caption_only', 'hybrid'],
                       help='Model type to train')
    parser.add_argument('--parameter_file', type=str, default='parameter.json',
                       help='Path to parameter file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Train model
    trainer = EnhancedTrainer(args.model_type, args.parameter_file)
    results = trainer.train(args.epochs)
    
    print(f"\nTraining completed for {args.model_type} model!")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main() 