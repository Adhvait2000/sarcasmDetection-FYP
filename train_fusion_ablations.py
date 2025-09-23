# train_late_sum.py
import argparse
import json
import torch

from train_enhanced import EnhancedTrainer
from fusion.late_sum_hybrid import HybridLateSumModel

class LateSumTrainer(EnhancedTrainer):
    def _initialize_model(self):
        if self.model_type != "hybrid":
            # defer to your existing mapping for other model types
            return super()._initialize_model()

        p = self.parameter
        return HybridLateSumModel(
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
            type_bmco=p["type_bmco"]
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['hybrid'], help='Late-sum ablation supports hybrid only')
    parser.add_argument('--parameter_file', type=str, default='parameter.json')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    trainer = LateSumTrainer(args.model_type, args.parameter_file)
    results = trainer.train(args.epochs)

    print(f"\n[late_sum] Completed for {args.model_type}")
    print(f"Best Val F1: {results['best_val_f1']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()
