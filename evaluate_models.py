import argparse
import json
import os
import torch
import torch.nn.functional as F
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

# UPDATED: import the new enhanced variants
from model_enhanced import (
    BaselineModel,                 # text+image baseline (no external knowledge branch)
    EnhancedKnowledgeOnlyModel,    # new knowledge-only model
    SuperiorHybridModel,           # new hybrid with tri-modal fusion
)

from utils.enhanced_dataset import EnhancedBaseSet, MultiKnowledgePadCollate
from utils.data_utils import construct_edge_image  # to build image graphs
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self, parameter_file="parameter.json"):
        """
        Model evaluator for comprehensive comparison
        """
        self.parameter_file = parameter_file

        with open(parameter_file) as f:
            self.parameter = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Map public names → classes. knowledge_types here are ONLY for dataset construction.
        self.model_configs = {
            "baseline": {
                "model_class": BaselineModel,
                "knowledge_types": [1],  # captions included in dataset (as text tokens)
                "description": "Image + Text (+captions as text; no external knowledge branch)",
            },
            "knowledge_only": {
                "model_class": EnhancedKnowledgeOnlyModel,
                "knowledge_types": [2, 3],  # ANPs + attributes
                "description": "Knowledge-only with attentive pooling & fusion (ANPs + Attributes)",
            },
            "hybrid": {
                "model_class": SuperiorHybridModel,
                "knowledge_types": [1, 2, 3],  # All knowledge types
                "description": "Tri-modal fusion: Image + Text + Captions + ANPs + Attributes",
            },
        }

    # ---------- helpers for device + image graph ----------
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
    # ------------------------------------------------------

    def load_model(self, model_type, checkpoint_path):
        """
        Load trained model from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        cfg = self.model_configs[model_type]
        ModelClass = cfg["model_class"]

        # Initialize with parameter.json fields; do NOT pass knowledge_types to constructors
        if model_type == "baseline":
            model = ModelClass(
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
                type_bmco=self.parameter["type_bmco"],
            )
        elif model_type == "knowledge_only":
            model = ModelClass(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                knowledge_types=[2, 3],
                max_knowledge_length=self.parameter.get("know_max_length", 20),
                cro_layers=self.parameter["cro_layers"],
                cro_heads=self.parameter["cro_heads"],
                cro_drop=self.parameter["cro_drop"],
                txt_gat_layer=self.parameter["txt_gat_layer"],
                txt_gat_drop=self.parameter["txt_gat_drop"],
                txt_gat_head=self.parameter["txt_gat_head"],
                lam=self.parameter["lambda"],
            )
        else:  # hybrid → SuperiorHybridModel
            model = ModelClass(
                txt_input_dim=self.parameter["txt_input_dim"],
                txt_out_size=self.parameter["txt_out_size"],
                img_input_dim=self.parameter["img_input_dim"],
                img_inter_dim=self.parameter["img_inter_dim"],
                img_out_dim=self.parameter["img_out_dim"],
                knowledge_types=[1, 2, 3],
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

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def create_data_loader(self, model_type, split="test"):
        """
        Create data loader for a specific model type/split.
        """
        knowledge_types = self.model_configs[model_type]["knowledge_types"]

        # dataset sampling as decimal
        dataset_percentage = self.parameter.get("dataset_percentage", 100.0)
        if dataset_percentage > 1.0:
            dataset_percentage = dataset_percentage / 100.0

        # image .pt files
        train_img_file = self.parameter.get("train_img_file", "train_B32.pt")
        val_img_file = self.parameter.get("val_img_file", "val_B32.pt")
        test_img_file = self.parameter.get("test_img_file", "test_B32.pt")
        split_to_file = {"train": train_img_file, "val": val_img_file, "test": test_img_file}

        dataset = EnhancedBaseSet(
            type=split,
            max_length=self.parameter["max_length"],
            text_path=os.path.join(self.parameter["annotation_files"], f"{split}.json"),
            img_path=os.path.join(self.parameter["DATA_DIR"], split_to_file[split]),
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
            dataset_percentage=dataset_percentage,
        )

        collate_fn = MultiKnowledgePadCollate(
            knowledge_types=knowledge_types,
            max_knowledge_length=self.parameter.get("know_max_length", 20),
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.parameter["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.parameter.get("num_workers", 4),
        )
        return data_loader

    def evaluate_model(self, model, data_loader, model_type):
        """
        Evaluate a single model on a given loader.
        """
        model.eval()
        all_predictions, all_probabilities, all_labels = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                if model_type == "knowledge_only":
                    (texts, mask_batch, t1_word_seq, txt_edge_index,
                     gnn_mask, np_mask, knowledge_inputs, knowledge_masks) = self._prepare_knowledge_only_batch(batch)

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
                    labels = batch[8].to(self.device)  # consistent indexing
                else:
                    (imgs, texts, mask_batch, img_edge_index, t1_word_seq,
                     txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks) = self._prepare_batch(batch)

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
                    labels = batch[8].to(self.device)

                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # numpy-ify
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)

        # metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )

        # AUC-ROC (binary)
        try:
            if all_probabilities.shape[1] >= 2:
                auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
            else:
                auc_roc = 0.5
        except Exception:
            auc_roc = 0.5

        cm = confusion_matrix(all_labels, all_predictions)
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
        """
        Prepare batch for baseline and hybrid models
        """
        imgs = batch[0].to(self.device)         # [B, K, D] where K = num_patches
        texts = batch[1]
        word_spans = batch[2]
        # word_len = batch[3]  # unused here
        mask_batch = batch[4].to(self.device)
        txt_edge_index = batch[5]
        gnn_mask = batch[6].to(self.device)
        np_mask = batch[7].to(self.device)

        # move tokenizer dicts
        texts = self._move_tokenizer_batch_to_device(texts)

        # build image edge_index for the whole batch: [B, 2, E]
        num_patches = imgs.size(1)
        img_edge_index = self._construct_image_edge_index(batch_size=imgs.size(0), num_patches=num_patches)

        # knowledge dicts
        knowledge_inputs, knowledge_masks = None, None
        if len(batch) > 9:
            knowledge_inputs, knowledge_masks = [], []
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
            img_edge_index,
            word_spans,
            txt_edge_index,
            gnn_mask,
            np_mask,
            knowledge_inputs,
            knowledge_masks,
        )

    def _prepare_knowledge_only_batch(self, batch):
        """
        Prepare batch for knowledge-only model
        """
        texts = batch[1]
        word_spans = batch[2]
        # word_len = batch[3]  # unused
        mask_batch = batch[4].to(self.device)
        txt_edge_index = batch[5]
        gnn_mask = batch[6].to(self.device)
        np_mask = batch[7].to(self.device)

        texts = self._move_tokenizer_batch_to_device(texts)

        knowledge_inputs, knowledge_masks = [], []
        if len(batch) > 9:
            for i in range(9, len(batch), 3):
                if batch[i] is not None:
                    knowledge_inputs.append(self._move_tokenizer_batch_to_device(batch[i]))
                    knowledge_masks.append(batch[i + 2].to(self.device) if batch[i + 2] is not None else None)
                else:
                    knowledge_inputs.append(None)
                    knowledge_masks.append(None)

        return texts, mask_batch, word_spans, txt_edge_index, gnn_mask, np_mask, knowledge_inputs, knowledge_masks

    def compare_models(self, checkpoint_paths):
        """
        Compare all model variants given their checkpoints
        """
        results = {}
        for model_type, ckpt in checkpoint_paths.items():
            print(f"\nEvaluating {model_type} model...")
            model = self.load_model(model_type, ckpt)
            data_loader = self.create_data_loader(model_type, "test")
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
        Save CSV/plots/markdown summary for the comparison
        """
        os.makedirs(output_dir, exist_ok=True)

        # table
        rows = []
        for model_type, res in results.items():
            rows.append(
                {
                    "Model": model_type,
                    "Description": self.model_configs[model_type]["description"],
                    "Accuracy": f"{res['accuracy']:.4f}",
                    "Precision": f"{res['precision']:.4f}",
                    "Recall": f"{res['recall']:.4f}",
                    "F1": f"{res['f1']:.4f}",
                    "AUC-ROC": f"{res['auc_roc']:.4f}",
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/model_comparison.csv", index=False)

        # plots
        self._create_comparison_plots(results, output_dir)

        with open(f"{output_dir}/detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        self._generate_summary_report(results, output_dir)
        print(f"\nEvaluation results saved to {output_dir}/")

    def _create_comparison_plots(self, results, output_dir):
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        model_names = list(results.keys())

        # bar charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            vals = [results[m][metric] for m in model_names]
            axes[i].bar(model_names, vals)
            axes[i].set_title(f"{metric.upper()} Comparison")
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis="x", rotation=45)
            for j, v in enumerate(vals):
                axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        axes[-1].remove()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # confusion matrices
        fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 6))
        if len(model_names) == 1:
            axes = [axes]
        for i, m in enumerate(model_names):
            cm = results[m]["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"{m} Confusion Matrix")
            axes[i].set_ylabel("True Label")
            axes[i].set_xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ROC curves (binary)
        plt.figure(figsize=(10, 8))
        for m in model_names:
            probs = results[m]["probabilities"]
            labels = results[m]["labels"]
            if probs.shape[1] >= 2:
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                auc = results[m]["auc_roc"]
                plt.plot(fpr, tpr, label=f"{m} (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_summary_report(self, results, output_dir):
        report = []
        report.append("# Model Comparison Summary Report")
        report.append("=" * 50)
        report.append("")

        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        for metric in metrics:
            best_model = max(results.keys(), key=lambda x: results[x][metric])
            best_value = results[best_model][metric]
            report.append(f"**Best {metric.upper()}**: {best_model} ({best_value:.4f})")

        report.append("\n## Detailed Results\n")
        for m, res in results.items():
            report.append(f"### {m.upper()}")
            report.append(f"**Description**: {self.model_configs[m]['description']}\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            for metric in metrics:
                report.append(f"| {metric.upper()} | {res[metric]:.4f} |")
            report.append("")

        with open(f"{output_dir}/summary_report.md", "w") as f:
            f.write("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Path to baseline model checkpoint")
    parser.add_argument("--knowledge_only_checkpoint", type=str, required=True, help="Path to knowledge-only model checkpoint")
    parser.add_argument("--hybrid_checkpoint", type=str, required=True, help="Path to hybrid model checkpoint")
    parser.add_argument("--parameter_file", type=str, default="parameter.json", help="Path to parameter file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.parameter_file)

    checkpoint_paths = {
        "baseline": args.baseline_checkpoint,
        "knowledge_only": args.knowledge_only_checkpoint,
        "hybrid": args.hybrid_checkpoint,
    }

    print("Starting model comparison...")
    results = evaluator.compare_models(checkpoint_paths)
    evaluator.generate_comparison_report(results, args.output_dir)
    print("\nModel comparison completed!")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
