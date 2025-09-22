"""
Enhanced Training Script with Sophisticated Fusion
Supports specialized learning rates and advanced model variants
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
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import enhanced models
from model_enhanced import (
    BaselineModel, 
    ImageOnlyModel, 
    EnhancedKnowledgeOnlyModel, 
    SuperiorHybridModel
)
from utils.enhanced_dataset import EnhancedBaseSet, MultiKnowledgePadCollate
from utils.data_utils import construct_edge_image, seed_everything
from utils.compute_scores import get_metrics, get_four_metrics

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
        Enhanced trainer for sophisticated model variants
        
        Args:
            model_type: "baseline", "knowledge_only", "hybrid", "image_only", or "enhanced_*"
            parameter_file: Path to parameter file
        """
        self.model_type = model_type
        
        # Load parameters
        with open(parameter_file) as f:
            self.parameter = json.load(f)
        
        # Initialize model based on type
        self.model = self._initialize_model()
        self.model.to(device=device)
        
        # Setup sophisticated optimizer with specialized learning rates
        self.optimizer = self._setup_specialized_optimizer()
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=self.parameter.get("patience", 3)
        )
        
        self.criterion = CrossEntropyLoss()
        
        # Initialize logger
        self.logger = Logger(f"logs/{model_type}_training")

        # Early Stopping parameters
        self.early_stopping_patience = self.parameter.get("early_stopping_patience", 5)
        self.early_stop_min_delta = self.parameter.get("early_stop_min_delta", 0.001)
        
    def _initialize_model(self):
        """Initialize model based on type with enhanced variants"""
        base_params = {
            "txt_input_dim": self.parameter["txt_input_dim"],
            "txt_out_size": self.parameter["txt_out_size"],
            "img_input_dim": self.parameter["img_input_dim"],
            "img_inter_dim": self.parameter["img_inter_dim"],
            "img_out_dim": self.parameter["img_out_dim"],
            "cro_layers": self.parameter["cro_layers"],
            "cro_heads": self.parameter["cro_heads"],
            "cro_drop": self.parameter["cro_drop"],
            "txt_gat_layer": self.parameter["txt_gat_layer"],
            "txt_gat_drop": self.parameter["txt_gat_drop"],
            "txt_gat_head": self.parameter["txt_gat_head"],
            "img_gat_layer": self.parameter["img_gat_layer"],
            "img_gat_drop": self.parameter["img_gat_drop"],
            "img_gat_head": self.parameter["img_gat_head"],
            "img_patch": self.parameter["img_patch"],
            "lam": self.parameter["lambda"],
            "type_bmco": self.parameter["type_bmco"]
        }
        
        if self.model_type == "baseline":
            return BaselineModel(**{k: v for k, v in base_params.items() if k != "knowledge_types"})
            
        elif self.model_type == "image_only":
            return ImageOnlyModel(
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                img_patch=self.parameter["img_patch"],
                drop=self.parameter.get("cro_drop", 0.5),
                lam=self.parameter["lambda"],
            )
            
        elif self.model_type == "enhanced_knowledge_only":
            return EnhancedKnowledgeOnlyModel(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                knowledge_types=[2, 3],  # ANP and attributes
                max_knowledge_length=self.parameter.get("know_max_length", 20),
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                lam=self.parameter["lambda"]
            )
            
        elif self.model_type == "superior_hybrid":
            return SuperiorHybridModel(
                **base_params,
                knowledge_types=[1, 2, 3],  # All knowledge types
                max_knowledge_length=self.parameter.get("know_max_length", 20)
            )
            
        elif self.model_type == "text_image":
            # Standard baseline without external knowledge
            params = base_params.copy()
            params.pop("knowledge_types", None)
            return BaselineModel(**params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _setup_specialized_optimizer(self):
        """
        Setup sophisticated optimizer with specialized learning rates
        Addresses the "equal LR & shared BERT" bottleneck
        """
        # Separate parameter groups for specialized learning
        text_bert_params = []
        knowledge_bert_params = []
        fusion_params = []
        image_params = []
        classifier_params = []
        attention_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Classify parameters by component
            if "txt_encoder.bert_model" in name or ("bert_model" in name and "knowledge" not in name):
                text_bert_params.append(param)
            elif "knowledge_encoder.knowledge_bert" in name or "knowledge_bert" in name:
                knowledge_bert_params.append(param)
            elif any(fusion_keyword in name for fusion_keyword in 
                    ["fusion", "cross_modal", "tri_modal", "knowledge_guided", "knowledge_pooling"]):
                fusion_params.append(param)
            elif any(img_keyword in name for img_keyword in 
                    ["img_encoder", "patch_", "image_", "conditioned"]):
                image_params.append(param)
            elif any(attn_keyword in name for attn_keyword in 
                    ["attention", "attn", "_attn", "multihead"]):
                attention_params.append(param)
            elif any(classifier_keyword in name for classifier_keyword in 
                    ["classifier", "output_layer", "linear1", "linear2"]):
                classifier_params.append(param)
            else:
                # Default to fusion params for unclassified parameters
                fusion_params.append(param)
        
        # Create parameter groups with specialized learning rates
        param_groups = []
        
        base_lr = self.parameter["lr"]
        
        # Text BERT: Standard learning rate (pre-trained, needs fine-tuning)
        if text_bert_params:
            param_groups.append({
                "params": text_bert_params, 
                "lr": base_lr,
                "name": "text_bert"
            })
        
        # Knowledge BERT: Lower learning rate (specialized processing)
        if knowledge_bert_params:
            param_groups.append({
                "params": knowledge_bert_params, 
                "lr": base_lr * 0.5,  # 50% of base LR
                "name": "knowledge_bert"
            })
        
        # Fusion modules: Higher learning rate (new components, need more learning)
        if fusion_params:
            param_groups.append({
                "params": fusion_params, 
                "lr": base_lr * 2.0,  # 200% of base LR
                "name": "fusion"
            })
        
        # Image processing: Moderate learning rate
        if image_params:
            param_groups.append({
                "params": image_params, 
                "lr": base_lr * 1.5,  # 150% of base LR
                "name": "image"
            })
        
        # Attention mechanisms: Higher learning rate (complex interactions)
        if attention_params:
            param_groups.append({
                "params": attention_params, 
                "lr": base_lr * 1.8,  # 180% of base LR
                "name": "attention"
            })
        
        # Classifiers: Highest learning rate (task-specific, needs rapid adaptation)
        if classifier_params:
            param_groups.append({
                "params": classifier_params, 
                "lr": base_lr * 3.0,  # 300% of base LR
                "name": "classifier"
            })
        
        # Fallback if no parameters were classified
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": base_lr}]
        
        # Print parameter group info
        print(f"\nOptimizer setup for {self.model_type}:")
        for i, group in enumerate(param_groups):
            group_name = group.get("name", f"group_{i}")
            param_count = sum(p.numel() for p in group["params"])
            print(f"  {group_name}: {param_count:,} params, LR: {group['lr']:.2e}")
        
        return optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.parameter.get("weight_decay", 0.01),  # Slightly higher weight decay
            amsgrad=True
        )
    
    def _get_knowledge_types(self):
        """Get knowledge types based on model type"""
        if self.model_type == "baseline":
            return [1]  # Only captions
        elif self.model_type in ["image_only", "text_image"]:
            return []  # No external knowledge
        elif self.model_type == "enhanced_knowledge_only":
            return [2, 3]  # ANP and attributes
        elif self.model_type == "superior_hybrid":
            return [1, 2, 3]  # All knowledge types
        else:
            return []

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

        # Datasets
        train_dataset = EnhancedBaseSet(
            type="train",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "train.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], train_img_file),
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=dataset_percentage,
        )

        val_dataset = EnhancedBaseSet(
            type="val",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "val.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], val_img_file),
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=1.0,
        )

        test_dataset = EnhancedBaseSet(
            type="test",
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], "test.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], test_img_file),
            knowledge_types=knowledge_types,
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
        """Construct image edge indices for the batch"""
        single_edge_index = construct_edge_image(num_patches)
        batch_edge_indices = []
        for b in range(batch_size):
            batch_edge_indices.append(single_edge_index)
        batch_tensor = torch.stack([single_edge_index for _ in range(batch_size)])
        return batch_tensor
        
    def train_epoch(self, train_loader):
        """Train for one epoch with enhanced monitoring"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Track losses by component for sophisticated models
        component_losses = {}
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Debug first batch
            if batch_idx == 0:
                print(f"\nFirst batch shapes:")
                print(f"  Images: {batch[0].shape if len(batch) > 0 else 'None'}")
                print(f"  Batch length: {len(batch)}")
            
            # Unpack batch data based on model type
            if "knowledge_only" in self.model_type:
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
                    knowledge_masks=knowledge_masks
                )
            elif self.model_type == "image_only":
                imgs = batch[0].to(device)
                outputs = self.model(imgs=imgs)
            else:
                # Enhanced models: baseline, superior_hybrid, text_image
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
                    knowledge_masks=knowledge_masks
                )
            
            # Get labels
            labels = batch[8].to(device)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # Enhanced gradient clipping
            max_grad_norm = self.parameter.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Gradient monitoring for sophisticated models
            if batch_idx % 100 == 0 and hasattr(self.model, 'classifier'):
                total_norm = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if batch_idx == 0:
                    print(f"Gradient norm: {total_norm:.4f}")
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def evaluate(self, data_loader, split="val"):
        """Evaluate model performance with enhanced metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                # Prepare batch based on model type
                if "knowledge_only" in self.model_type:
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
                        knowledge_masks=knowledge_masks
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
                        knowledge_masks=knowledge_masks
                    )
                
                # Get labels
                labels = batch[8].to(device)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Store predictions and confidence scores
                probabilities = torch.softmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Compute enhanced metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Confidence-based metrics
        avg_confidence = np.mean(all_confidences)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': avg_confidence,
            'confusion_matrix': cm
        }
    
    def _prepare_batch(self, batch):
        """Prepare batch for enhanced models"""
        imgs = batch[0].to(device)
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]
        mask_batch = batch[4].to(device)
        txt_edge_index = batch[5]
        gnn_mask = batch[6].to(device)
        np_mask = batch[7].to(device)
        
        # Construct proper image edge indices
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
        """Save model checkpoint with enhanced metadata"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'model_type': self.model_type,
            'parameter': self.parameter,
            'model_architecture': str(self.model),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        path = f"{save_dir}/{self.model_type}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        return path
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save enhanced confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_type.title()}', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, num_epochs):
        """Enhanced training loop with sophisticated monitoring"""
        print(f"\nTraining {self.model_type} model with sophisticated fusion...")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Batch size: {self.parameter['batch_size']}")
        print(f"Base learning rate: {self.parameter['lr']}")
        print(f"Epochs: {num_epochs}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()
        
        best_val_f1 = float('-inf')
        best_epoch = -1
        best_ckpt_path = None
        epochs_no_improve = 0
        
        # Training metrics tracking
        training_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 
            'val_acc': [], 'val_f1': [], 'val_confidence': []
        }
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
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
            self.logger.log_scalar('val_confidence', val_metrics['avg_confidence'], epoch)
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['val_f1'].append(val_metrics['f1'])
            training_history['val_confidence'].append(val_metrics['avg_confidence'])
            
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, Val Confidence: {val_metrics['avg_confidence']:.4f}")
            
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
        
        # Plot training history
        self._plot_training_history(training_history)
        
        # Load best model for testing
        if best_ckpt_path:
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {best_epoch+1}")
        
        # Test evaluation
        print(f"\n{'='*60}")
        print(f"Testing best model (epoch {best_epoch+1})")
        print(f"{'='*60}")
        
        test_metrics = self.evaluate(test_loader, "test")
        
        print(f"\nTest Results:")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Precision:   {test_metrics['precision']:.4f}")
        print(f"  Recall:      {test_metrics['recall']:.4f}")
        print(f"  F1 Score:    {test_metrics['f1']:.4f}")
        print(f"  Confidence:  {test_metrics['avg_confidence']:.4f}")
        
        # Save enhanced results
        results = {
            'model_type': self.model_type,
            'model_architecture': 'sophisticated_fusion',
            'best_epoch': best_epoch + 1,
            'best_val_f1': best_val_f1,
            'training_history': training_history,
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'avg_confidence': float(test_metrics['avg_confidence'])
            },
            'model_stats': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        results_file = f"results_{self.model_type}_enhanced.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEnhanced results saved to: {results_file}")
        
        return results
    
    def _plot_training_history(self, history):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Train Acc', color='green')
        axes[0, 1].plot([acc * 100 for acc in history['val_acc']], label='Val Acc', color='orange')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(history['val_f1'], label='Val F1', color='purple')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confidence
        axes[1, 1].plot(history['val_confidence'], label='Val Confidence', color='brown')
        axes[1, 1].set_title('Model Confidence')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Average Confidence')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Training History - {self.model_type.title()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'enhanced_knowledge_only', 'superior_hybrid', 
                               'image_only', 'text_image'],
                       help='Enhanced model type to train')
    parser.add_argument('--parameter_file', type=str, default='parameter.json',
                       help='Path to parameter file')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Train enhanced model
    trainer = EnhancedTrainer(args.model_type, args.parameter_file)
    results = trainer.train(args.epochs)
    
    print(f"\nSophisticated training completed for {args.model_type} model!")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")
    print(f"Model parameters: {results['model_stats']['total_parameters']:,}")

if __name__ == "__main__":
    main()