# train_late_sum.py  (now supports both late_sum & logit_gate)
import argparse
from train_enhanced import EnhancedTrainer
from fusion.late_sum_hybrid import HybridLateSumModel
from fusion.logit_gate_hybrid import HybridLogitGateModel
from fusion.attn_weighted_hybrid import HybridAttnWeightedModel
from utils.logging.tf_logger import SimpleLogger as Logger

class FusionAblationTrainer(EnhancedTrainer):
    def __init__(self, model_type="hybrid", parameter_file="parameter.json", fusion="late_sum"):
        self._fusion_variant = fusion
        super().__init__(model_type=model_type, parameter_file=parameter_file)

        self.logger = Logger(f"logs/{model_type}_{fusion}_training")

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

        if self._fusion_variant == "late_sum":
            return HybridLateSumModel(**common_kwargs)
        elif self._fusion_variant == "logit_gate":
            return HybridLogitGateModel(**common_kwargs)
        elif self._fusion_variant == "attn_weighted":
            return HybridAttnWeightedModel(**common_kwargs)   
        else:
            raise ValueError(f"Unknown fusion variant: {self._fusion_variant}")

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
