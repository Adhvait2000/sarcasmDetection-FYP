"""
Enhanced Training Script
Supports all three model variants with comprehensive evaluation
"""
import argparse
from tqdm import tqdm
import os
from utils.logging.tf_logger import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_enhanced import BaselineModel, KnowledgeOnlyModel, HybridModel
from utils.enhanced_dataset import EnhancedBaseSet, MultiKnowledgePadCollate
from utils.data_utils import construct_edge_image, seed_everything
from utils.compute_scores import get_metrics, get_four_metrics

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.multiprocessing.set_sharing_strategy('file_system')

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
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.parameter["lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.parameter["weight_decay"],
            amsgrad=True
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=self.parameter["patience"],
            verbose=True
        )
        
        self.criterion = CrossEntropyLoss()
        
        # Initialize logger
        self.logger = Logger(f"logs/{model_type}_training")
        
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
        elif self.model_type == "knowledge_only":
            return KnowledgeOnlyModel(
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
        """Get knowledge types based on model type"""
        if self.model_type == "baseline":
            return [1]  # Only captions
        elif self.model_type == "knowledge_only":
            return [2, 3]  # ANP and attributes
        elif self.model_type == "hybrid":
            return [1, 2, 3]  # All knowledge types
        else:
            return [0]  # No knowledge
    
    def _create_data_loaders(self):
        """Create data loaders for the specific model type"""
        knowledge_types = self._get_knowledge_types()
        
        # Create datasets
        train_dataset = EnhancedBaseSet(
            type="train",
            max_length=self.parameter["max_length"],
            text_path=self.parameter["annotation_files"] + "/train.json",
            img_path=self.parameter["DATA_DIR"] + "/train.pt",
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20)
        )
        
        val_dataset = EnhancedBaseSet(
            type="val",
            max_length=self.parameter["max_length"],
            text_path=self.parameter["annotation_files"] + "/val.json",
            img_path=self.parameter["DATA_DIR"] + "/val.pt",
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20)
        )
        
        test_dataset = EnhancedBaseSet(
            type="test",
            max_length=self.parameter["max_length"],
            text_path=self.parameter["annotation_files"] + "/test.json",
            img_path=self.parameter["DATA_DIR"] + "/test.pt",
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20)
        )
        
        # Create collate function
        collate_fn = MultiKnowledgePadCollate(
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Unpack batch data
            if self.model_type == "knowledge_only":
                # Knowledge-only model doesn't use images
                texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, np_mask, \
                knowledge_inputs, knowledge_masks = self._prepare_knowledge_only_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    texts=texts,
                    mask_batch=mask_batch,
                    t1_word_seq=t1_word_seq,
                    txt_edge_index=txt_edge_index,
                    gnn_mask=gnn_mask,
                    np_mask=np_mask,
                    knowledge_inputs=knowledge_inputs,
                    knowledge_masks=knowledge_masks
                )
            else:
                # Baseline and hybrid models use images
                imgs, texts, mask_batch, img_edge_index, t1_word_seq, txt_edge_index, \
                gnn_mask, np_mask, knowledge_inputs, knowledge_masks = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    imgs=imgs,
                    texts=texts,
                    mask_batch=mask_batch,
                    img_edge_index=img_edge_index,
                    t1_word_seq=t1_word_seq,
                    txt_edge_index=txt_edge_index,
                    gnn_mask=gnn_mask,
                    np_mask=np_mask,
                    knowledge_inputs=knowledge_inputs,
                    knowledge_masks=knowledge_masks
                )
            
            # Get labels
            labels = batch[-1] if self.model_type == "knowledge_only" else batch[8]
            labels = labels.to(device)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                # Prepare batch
                if self.model_type == "knowledge_only":
                    texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, np_mask, \
                    knowledge_inputs, knowledge_masks = self._prepare_knowledge_only_batch(batch)
                    
                    outputs = self.model(
                        texts=texts,
                        mask_batch=mask_batch,
                        t1_word_seq=t1_word_seq,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks
                    )
                else:
                    imgs, texts, mask_batch, img_edge_index, t1_word_seq, txt_edge_index, \
                    gnn_mask, np_mask, knowledge_inputs, knowledge_masks = self._prepare_batch(batch)
                    
                    outputs = self.model(
                        imgs=imgs,
                        texts=texts,
                        mask_batch=mask_batch,
                        img_edge_index=img_edge_index,
                        t1_word_seq=t1_word_seq,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks
                    )
                
                # Get labels
                labels = batch[-1] if self.model_type == "knowledge_only" else batch[8]
                labels = labels.to(device)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def _prepare_batch(self, batch):
        """Prepare batch for baseline and hybrid models"""
        imgs = batch[0].to(device)
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]
        mask_batch = batch[4].to(device)
        edge_cap1 = batch[5]
        gnn_mask_1 = batch[6].to(device)
        np_mask_1 = batch[7].to(device)
        labels = batch[8]
        
        # Handle knowledge data
        knowledge_inputs = None
        knowledge_masks = None
        
        if len(batch) > 9:
            knowledge_inputs = []
            knowledge_masks = []
            
            # Extract knowledge inputs and masks
            for i in range(9, len(batch), 3):
                if i < len(batch) and batch[i] is not None:
                    knowledge_inputs.append(batch[i])
                    knowledge_masks.append(batch[i+2].to(device) if batch[i+2] is not None else None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)
        
        return imgs, texts, mask_batch, edge_cap1, word_spans, edge_cap1, \
               gnn_mask_1, np_mask_1, knowledge_inputs, knowledge_masks
    
    def _prepare_knowledge_only_batch(self, batch):
        """Prepare batch for knowledge-only model"""
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]
        mask_batch = batch[4].to(device)
        edge_cap1 = batch[5]
        gnn_mask_1 = batch[6].to(device)
        np_mask_1 = batch[7].to(device)
        
        # Handle knowledge data
        knowledge_inputs = []
        knowledge_masks = []
        
        if len(batch) > 9:
            for i in range(9, len(batch), 3):
                if i < len(batch) and batch[i] is not None:
                    knowledge_inputs.append(batch[i])
                    knowledge_masks.append(batch[i+2].to(device) if batch[i+2] is not None else None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)
        
        return texts, mask_batch, word_spans, edge_cap1, gnn_mask_1, np_mask_1, \
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
        
        torch.save(checkpoint, f"{save_dir}/{self.model_type}_epoch_{epoch}.pt")
    
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
        print(f"Training {self.model_type} model...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()
        
        best_val_f1 = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
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
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                self.save_model(epoch, val_metrics)
                
                # Plot confusion matrix
                self.plot_confusion_matrix(
                    val_metrics['confusion_matrix'],
                    f"confusion_matrix_{self.model_type}_epoch_{epoch}.png"
                )
        
        # Test on best model
        print(f"\nTesting best model (epoch {best_epoch})...")
        test_metrics = self.evaluate(test_loader, "test")
        
        print(f"Test Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        
        # Save final results
        results = {
            'model_type': self.model_type,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics
        }
        
        with open(f"results_{self.model_type}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'knowledge_only', 'hybrid'],
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