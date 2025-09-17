import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import warnings

warnings.filterwarnings("ignore")

from model_enhanced import BaselineModel, KnowledgeOnlyModel, HybridModel
from utils.enhanced_dataset import EnhancedBaseSet, MultiKnowledgePadCollate
from utils.data_utils import construct_edge_image  # ✅ needed to build image graphs
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self, parameter_file="parameter.json"):
        """
        Model evaluator for comprehensive comparison

        Args:
            parameter_file: Path to parameter file
        """
        self.parameter_file = parameter_file

        # Load parameters
        with open(parameter_file) as f:
            self.parameter = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        self.model_configs = {
            "baseline": {
                "model_class": BaselineModel,
                "knowledge_types": [1],  # Only captions
                "description": "Image + Text + Captions",
            },
            "knowledge_only": {
                "model_class": KnowledgeOnlyModel,
                "knowledge_types": [2, 3],  # ANP + attributes
                "description": "ANPs + Attributes (no image patches)",
            },
            "hybrid": {
                "model_class": HybridModel,
                "knowledge_types": [1, 2, 3],  # All knowledge types
                "description": "Image + Text + Captions + ANPs + Attributes",
            },
        }

    # ---------- NEW: helpers for device + image graph ----------
    def _move_tokenizer_batch_to_device(self, batch_dict):
        """Move a HuggingFace tokenizer dict (input_ids, attention_mask, etc.) to self.device."""
        if batch_dict is None:
            return None
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch_dict.items()}

    def _construct_image_edge_index(self, batch_size, num_patches):
        """
        Build a batched image graph tensor of shape [B, 2, E].
        `construct_edge_image(num_patches)` returns a single [2, E] tensor.
        We stack it for each batch element.
        """
        single = construct_edge_image(num_patches)  # [2, E]
        single = single.to(self.device)
        return torch.stack([single for _ in range(batch_size)], dim=0)  # [B, 2, E]
    # -----------------------------------------------------------

    def load_model(self, model_type, checkpoint_path):
        """
        Load trained model from checkpoint

        Args:
            model_type: Type of model to load
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model
        model_config = self.model_configs[model_type]
        model = model_config["model_class"](
            txt_input_dim=self.parameter["txt_input_dim"],
            txt_out_size=self.parameter["txt_out_size"],
            img_input_dim=self.parameter["img_input_dim"],
            img_inter_dim=self.parameter["img_inter_dim"],
            img_out_dim=self.parameter["img_out_dim"],
            knowledge_types=model_config["knowledge_types"],
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
            type_bmco=self.parameter["type_bmco"],
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def create_data_loader(self, model_type, split="test"):
        """
        Create data loader for specific model type

        Args:
            model_type: Type of model
            split: Data split to use

        Returns:
            DataLoader
        """
        model_config = self.model_configs[model_type]
        knowledge_types = model_config["knowledge_types"]

        # Create dataset with dataset sampling
        dataset_percentage = self.parameter.get("dataset_percentage", 100.0) / 100.0  # Convert percentage to decimal

        # Get file names from parameters or use defaults
        train_img_file = self.parameter.get("train_img_file", "train_B32.pt")
        val_img_file = self.parameter.get("val_img_file", "val_B32.pt")
        test_img_file = self.parameter.get("test_img_file", "test_B32.pt")

        # Map split to correct file
        split_to_file = {"train": train_img_file, "val": val_img_file, "test": test_img_file}

        dataset = EnhancedBaseSet(
            type=split,
            max_length=self.parameter["max_length"],
            text_path=self.parameter["annotation_files"] + f"/{split}.json",
            img_path=self.parameter["DATA_DIR"] + "/" + split_to_file[split],
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=dataset_percentage,
        )

        # Create collate function
        collate_fn = MultiKnowledgePadCollate(
            knowledge_types=knowledge_types, max_knowledge_length=self.parameter.get("know_max_length", 20)
        )

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

        return data_loader

    def evaluate_model(self, model, data_loader, model_type):
        """
        Evaluate a single model

        Args:
            model: Model to evaluate
            data_loader: Data loader
            model_type: Type of model

        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # Prepare batch
                if model_type == "knowledge_only":
                    texts, mask_batch, t1_word_seq, txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks = (
                        self._prepare_knowledge_only_batch(batch)
                    )

                    outputs = model(
                        texts=texts,
                        mask_batch=mask_batch,
                        t1_word_seq=t1_word_seq,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks,
                    )
                else:
                    (
                        imgs,
                        texts,
                        mask_batch,
                        img_edge_index,
                        t1_word_seq,
                        txt_edge_index,
                        gnn_mask,
                        np_mask,
                        knowledge_inputs,
                        knowledge_masks,
                    ) = self._prepare_batch(batch)

                    outputs = model(
                        imgs=imgs,
                        texts=texts,
                        mask_batch=mask_batch,
                        img_edge_index=img_edge_index,
                        t1_word_seq=t1_word_seq,
                        txt_edge_index=txt_edge_index,
                        gnn_mask=gnn_mask,
                        np_mask=np_mask,
                        knowledge_inputs=knowledge_inputs,
                        knowledge_masks=knowledge_masks,
                    )

                # Get labels (same indexing pattern as in training code)
                labels = batch[-1] if model_type == "knowledge_only" else batch[8]
                labels = labels.to(self.device)

                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="weighted")

        # Compute AUC-ROC (binary)
        try:
            auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except Exception:
            auc_roc = 0.5  # Default for binary classification

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Generate classification report
        class_report = classification_report(all_labels, all_predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "predictions": all_predictions,
            "probabilities": all_probabilities,
            "labels": all_labels,
        }

    def _prepare_batch(self, batch):
        """Prepare batch for baseline and hybrid models"""
        imgs = batch[0].to(self.device)  # [B, K, D] where K = num_patches
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]  # (unused here)
        mask_batch = batch[4].to(self.device)
        txt_edge_index = batch[5]  # text graph (list/seq per sample or batched form)
        gnn_mask_1 = batch[6].to(self.device)
        np_mask_1 = batch[7].to(self.device)

        # ✅ move tokenizer dicts to device
        texts = self._move_tokenizer_batch_to_device(texts)

        # ✅ build proper image edge_index for the whole batch: [B, 2, E]
        img_edge_index = self._construct_image_edge_index(batch_size=imgs.size(0), num_patches=imgs.size(1))

        # Handle knowledge data (move dicts to device)
        knowledge_inputs = None
        knowledge_masks = None
        if len(batch) > 9:
            knowledge_inputs = []
            knowledge_masks = []
            for i in range(9, len(batch), 3):
                if batch[i] is not None:
                    knowledge_inputs.append(self._move_tokenizer_batch_to_device(batch[i]))
                    knowledge_masks.append(batch[i + 2].to(self.device) if batch[i + 2] is not None else None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)

        return (
            imgs,
            texts,
            mask_batch,
            img_edge_index,  # image graph here
            word_spans,
            txt_edge_index,  # text graph here
            gnn_mask_1,
            np_mask_1,
            knowledge_inputs,
            knowledge_masks,
        )

    def _prepare_knowledge_only_batch(self, batch):
        """Prepare batch for knowledge-only model"""
        texts = batch[1]
        word_spans = batch[2]
        word_len = batch[3]  # (unused)
        mask_batch = batch[4].to(self.device)
        txt_edge_index = batch[5]
        gnn_mask_1 = batch[6].to(self.device)
        np_mask_1 = batch[7].to(self.device)

        # ✅ move tokenizer dicts to device
        texts = self._move_tokenizer_batch_to_device(texts)

        # Handle knowledge data (move dicts to device)
        knowledge_inputs = []
        knowledge_masks = []
        if len(batch) > 9:
            for i in range(9, len(batch), 3):
                if batch[i] is not None:
                    knowledge_inputs.append(self._move_tokenizer_batch_to_device(batch[i]))
                    knowledge_masks.append(batch[i + 2].to(self.device) if batch[i + 2] is not None else None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)

        return texts, mask_batch, word_spans, txt_edge_index, gnn_mask_1, np_mask_1, knowledge_inputs, knowledge_masks

    def compare_models(self, checkpoint_paths):
        """
        Compare all model variants

        Args:
            checkpoint_paths: Dictionary mapping model types to checkpoint paths

        Returns:
            Comparison results
        """
        results = {}

        for model_type, checkpoint_path in checkpoint_paths.items():
            print(f"\nEvaluating {model_type} model...")

            # Load model
            model = self.load_model(model_type, checkpoint_path)

            # Create data loader
            data_loader = self.create_data_loader(model_type, "test")

            # Evaluate model
            model_results = self.evaluate_model(model, data_loader, model_type)

            results[model_type] = model_results

            print(f"{model_type} Results:")
            print(f"  Accuracy:  {model_results['accuracy']:.4f}")
            print(f"  Precision: {model_results['precision']:.4f}")
            print(f"  Recall:    {model_results['recall']:.4f}")
            print(f"  F1:        {model_results['f1']:.4f}")
            print(f"  AUC-ROC:   {model_results['auc_roc']:.4f}")

        return results

    def generate_comparison_report(self, results, output_dir="evaluation_results"):
        """
        Generate comprehensive comparison report

        Args:
            results: Evaluation results for all models
            output_dir: Output directory for reports
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create comparison table
        comparison_data = []
        for model_type, result in results.items():
            comparison_data.append(
                {
                    "Model": model_type,
                    "Description": self.model_configs[model_type]["description"],
                    "Accuracy": f"{result['accuracy']:.4f}",
                    "Precision": f"{result['precision']:.4f}",
                    "Recall": f"{result['recall']:.4f}",
                    "F1": f"{result['f1']:.4f}",
                    "AUC-ROC": f"{result['auc_roc']:.4f}",
                }
            )

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Save comparison table
        df.to_csv(f"{output_dir}/model_comparison.csv", index=False)

        # Create visualization
        self._create_comparison_plots(results, output_dir)

        # Save detailed results
        with open(f"{output_dir}/detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate summary report
        self._generate_summary_report(results, output_dir)

        print(f"\nEvaluation results saved to {output_dir}/")

    def _create_comparison_plots(self, results, output_dir):
        """Create comparison plots"""
        # Metrics comparison
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        model_names = list(results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[i].bar(model_names, values)
            axes[i].set_title(f"{metric.upper()} Comparison")
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Remove extra subplot
        axes[-1].remove()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, model_type in enumerate(model_names):
            cm = results[model_type]["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"{model_type} Confusion Matrix")
            axes[i].set_ylabel("True Label")
            axes[i].set_xlabel("Predicted Label")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ROC curves
        plt.figure(figsize=(10, 8))

        for model_type in model_names:
            probabilities = results[model_type]["probabilities"]
            labels = results[model_type]["labels"]

            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc = results[model_type]["auc_roc"]

            plt.plot(fpr, tpr, label=f"{model_type} (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_summary_report(self, results, output_dir):
        """Generate summary report"""
        report = []
        report.append("# Model Comparison Summary Report")
        report.append("=" * 50)
        report.append("")

        # Find best model for each metric
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

        for metric in metrics:
            best_model = max(results.keys(), key=lambda x: results[x][metric])
            best_value = results[best_model][metric]
            report.append(f"**Best {metric.upper()}**: {best_model} ({best_value:.4f})")

        report.append("")
        report.append("## Detailed Results")
        report.append("")

        for model_type, result in results.items():
            report.append(f"### {model_type.upper()}")
            report.append(f"**Description**: {self.model_configs[model_type]['description']}")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            for metric in metrics:
                report.append(f"| {metric.upper()} | {result[metric]:.4f} |")
            report.append("")

        # Save report
        with open(f"{output_dir}/summary_report.md", "w") as f:
            f.write("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_checkpoint", type=str, required=True, help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--knowledge_only_checkpoint", type=str, required=True, help="Path to knowledge-only model checkpoint"
    )
    parser.add_argument(
        "--hybrid_checkpoint", type=str, required=True, help="Path to hybrid model checkpoint"
    )
    parser.add_argument("--parameter_file", type=str, default="parameter.json", help="Path to parameter file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.parameter_file)

    # Define checkpoint paths
    checkpoint_paths = {
        "baseline": args.baseline_checkpoint,
        "knowledge_only": args.knowledge_only_checkpoint,
        "hybrid": args.hybrid_checkpoint,
    }

    # Compare models
    print("Starting model comparison...")
    results = evaluator.compare_models(checkpoint_paths)

    # Generate report
    evaluator.generate_comparison_report(results, args.output_dir)

    print("\nModel comparison completed!")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
