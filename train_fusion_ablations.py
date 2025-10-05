# train_late_sum.py  (now supports both late_sum & logit_gate)
import argparse
import os
import json
from train_enhanced import EnhancedTrainer
from fusion.mid_linear_hybrid import HybridMidLinearModel
from fusion.logit_gate_hybrid import HybridLogitGateModel
from fusion.early_fusion_hybrid import FiLMEarlyFusion
from utils.logging.tf_logger import SimpleLogger as Logger

class FusionAblationTrainer(EnhancedTrainer):
    def __init__(self, model_type="hybrid", parameter_file="parameter.json", fusion="late_sum"):
        self._fusion_variant = fusion
        super().__init__(model_type=model_type, parameter_file=parameter_file)

        self.logger = Logger(f"logs/{model_type}_{fusion}_training")

    def _fusion_suffix(self):
        return f"_{self._fusion_variant}" if getattr(self, "_fusion_variant", None) else ""


    def _initialize_model(self):
        # For non-hybrid, use your original mapping
        if self.model_type != "hybrid":
            return super()._initialize_model()

        p = self.parameter
        common_kwargs = dict(
            txt_input_dim=p["txt_input_dim"],
            txt_out_size=p["txt_out_size"],
            img_input_dim=p["img_input_dim"],
            img_inter_dim=p["img_inter_dim"],
            img_out_dim=p["img_out_dim"],
            knowledge_types=[1, 2, 3],
            max_knowledge_length=p.get("know_max_length", 20),
            cro_layers=p["cro_layers"],
            cro_heads=p["cro_heads"],
            cro_drop=p["cro_drop"],
            txt_gat_layer=p["txt_gat_layer"],
            txt_gat_drop=p["txt_gat_drop"],
            txt_gat_head=p["txt_gat_head"],
            img_gat_layer=p["img_gat_layer"],
            img_gat_drop=p["img_gat_drop"],
            img_gat_head=p["img_gat_head"],
            img_patch=p["img_patch"],
            lam=p["lambda"],
            type_bmco=p["type_bmco"],
        )

        if self._fusion_variant == "mid_linear":
            return HybridMidLinearModel(**common_kwargs)
        elif self._fusion_variant == "logit_gate":
            return HybridLogitGateModel(**common_kwargs)
        elif self._fusion_variant == "early_film":
            return FiLMEarlyFusion(**common_kwargs)   
        else:
            raise ValueError(f"Unknown fusion variant: {self._fusion_variant}")
        
    # --- override filenames for checkpoints ---
    def save_model(self, epoch, metrics, save_dir="saved_models"):
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
        path = f"{save_dir}/{self.model_type}{self._fusion_suffix()}_epoch_{epoch}.pt"
        import torch
        torch.save(checkpoint, path)
        return path
    
     # --- ensure confusion matrix filename includes fusion ---
    def plot_confusion_matrix(self, cm, save_path):
        # inject fusion suffix into whatever path parent passes
        root, ext = os.path.splitext(save_path)
        fused_path = f"{root}{self._fusion_suffix()}{ext}"
        return super().plot_confusion_matrix(cm, fused_path)
    
    # --- after training, also emit a fusion-suffixed results file ---
    def train(self, num_epochs):
        results = super().train(num_epochs)

        # Write an extra results file with fusion suffix to avoid collisions
        results_file = f"results_{self.model_type}{self._fusion_suffix()}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[Info] Also saved fusion-specific results to: {results_file}")

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['hybrid'])
    parser.add_argument('--fusion', type=str, default='late_sum',
                        choices=['late_sum', 'logit_gate', 'attn_weighted']) 
    parser.add_argument('--parameter_file', type=str, default='parameter.json')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    trainer = FusionAblationTrainer(args.model_type, args.parameter_file, fusion=args.fusion)
    results = trainer.train(args.epochs)

    print(f"\n[{args.fusion}] Completed for {args.model_type}")
    print(f"Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()
