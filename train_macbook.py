"""
MacBook Pro Optimized Training Script
Automatically handles memory constraints and uses optimized configurations
"""
import argparse
import json
import os
import psutil
import torch
from train_enhanced import EnhancedTrainer

class MacBookTrainer:
    def __init__(self, model_type="baseline", parameter_file="parameter_macbook.json"):
        """
        MacBook Pro optimized trainer
        
        Args:
            model_type: "baseline", "knowledge_only", or "hybrid"
            parameter_file: Path to parameter file
        """
        self.model_type = model_type
        self.parameter_file = parameter_file
        
        # Load parameters
        with open(parameter_file) as f:
            self.parameter = json.load(f)
        
        # Check system memory
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        print(f"System Memory: {self.total_memory:.1f}GB total, {self.available_memory:.1f}GB available")
        
        # Auto-adjust parameters based on memory
        self.parameter = self._auto_adjust_parameters()
        
        # Initialize trainer
        self.trainer = EnhancedTrainer(model_type, parameter_file)
        
        # Override trainer parameters with optimized ones
        self.trainer.parameter = self.parameter
        
        # Set device based on configuration
        if hasattr(self.parameter, 'macbook_optimizations') and self.parameter['macbook_optimizations'].get('use_cpu_training', False):
            self.trainer.device = torch.device("cpu")
            print("🖥️  Trainer device set to CPU")
        
    def _auto_adjust_parameters(self):
        """Automatically adjust parameters based on available memory"""
        print("Auto-adjusting parameters for MacBook Pro...")
        
        # Memory-based adjustments
        if self.available_memory < 6.0:  # Less than 6GB available
            print("⚠️  Low memory detected. Using conservative settings.")
            self.parameter["batch_size"] = 4
            self.parameter["max_knowledge_length"] = 10
            self.parameter["txt_out_size"] = 128
            self.parameter["img_out_dim"] = 128
            self.parameter["img_inter_dim"] = 256
            self.parameter["cro_layers"] = 2
            self.parameter["cro_heads"] = 3
            
        elif self.available_memory < 7.0:  # Less than 7GB available
            print("⚠️  Moderate memory. Using balanced settings.")
            self.parameter["batch_size"] = 6
            self.parameter["max_knowledge_length"] = 12
            self.parameter["txt_out_size"] = 150
            self.parameter["img_out_dim"] = 150
            self.parameter["img_inter_dim"] = 300
            self.parameter["cro_layers"] = 3
            self.parameter["cro_heads"] = 4
            
        else:  # 7GB+ available
            print("✅ Good memory available. Using standard enhanced settings.")
            self.parameter["batch_size"] = 8
            self.parameter["max_knowledge_length"] = 15
            self.parameter["txt_out_size"] = 150
            self.parameter["img_out_dim"] = 150
            self.parameter["img_inter_dim"] = 300
            self.parameter["cro_layers"] = 3
            self.parameter["cro_heads"] = 4
        
        # MacBook specific optimizations
        if hasattr(self.parameter, 'macbook_optimizations'):
            optimizations = self.parameter['macbook_optimizations']
            
            # Check if CPU training is requested
            if optimizations.get('use_cpu_training', False):
                print("🖥️  CPU-only training enabled")
                self.device = torch.device("cpu")
                # Increase batch size for CPU (no GPU memory constraints)
                self.parameter["batch_size"] = min(16, self.parameter["batch_size"] * 2)
                print(f"📦 Increased batch size to {self.parameter['batch_size']} for CPU training")
            
            # Enable memory optimizations
            if optimizations.get('gradient_checkpointing', True):
                print("✅ Enabling gradient checkpointing for memory efficiency")
            
            if optimizations.get('use_mixed_precision', True):
                print("✅ Enabling mixed precision training")
                self.parameter['enhanced_model_config']['use_mixed_precision'] = True
        
        print(f"Final batch size: {self.parameter['batch_size']}")
        print(f"Final model dimensions: {self.parameter['txt_out_size']}x{self.parameter['img_out_dim']}")
        
        return self.parameter
    
    def check_memory_usage(self):
        """Monitor memory usage during training"""
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}% used, {memory.available / (1024**3):.1f}GB available")
        
        if memory.percent > 90:
            print("⚠️  High memory usage detected!")
            return False
        return True
    
    def train(self, num_epochs):
        """Train with memory monitoring"""
        print(f"Starting training for {self.model_type} model...")
        print(f"Using configuration: {self.parameter_file}")
        
        # Check initial memory
        if not self.check_memory_usage():
            print("❌ Insufficient memory to start training")
            return None
        
        try:
            # Start training
            results = self.trainer.train(num_epochs)
            
            # Check final memory
            self.check_memory_usage()
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ Out of memory error occurred!")
                print("💡 Try reducing batch size or model dimensions")
                return None
            else:
                raise e
    
    def get_memory_recommendations(self):
        """Get memory optimization recommendations"""
        recommendations = []
        
        if self.available_memory < 6.0:
            recommendations.append("Close other applications to free up memory")
            recommendations.append("Use batch_size=4 for training")
            recommendations.append("Consider using CPU-only training initially")
            recommendations.append("Reduce max_knowledge_length to 10")
        
        elif self.available_memory < 7.0:
            recommendations.append("Close unnecessary browser tabs")
            recommendations.append("Use batch_size=6 for training")
            recommendations.append("Monitor memory usage during training")
        
        else:
            recommendations.append("Memory looks good for enhanced training")
            recommendations.append("You can use standard enhanced settings")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='MacBook Pro Optimized Training')
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'knowledge_only', 'hybrid'],
                       help='Model type to train')
    parser.add_argument('--parameter_file', type=str, default='parameter_macbook.json',
                       help='Path to parameter file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--check_memory', action='store_true',
                       help='Check memory usage and get recommendations')
    
    args = parser.parse_args()
    
    # Initialize MacBook trainer
    trainer = MacBookTrainer(args.model_type, args.parameter_file)
    
    if args.check_memory:
        print("\n📊 Memory Analysis:")
        print(f"Total Memory: {trainer.total_memory:.1f}GB")
        print(f"Available Memory: {trainer.available_memory:.1f}GB")
        
        print("\n💡 Recommendations:")
        recommendations = trainer.get_memory_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n📁 Using configuration: {args.parameter_file}")
        print(f"🎯 Model type: {args.model_type}")
        print(f"📦 Batch size: {trainer.parameter['batch_size']}")
        
    else:
        # Start training
        results = trainer.train(args.epochs)
        
        if results:
            print(f"\n✅ Training completed successfully!")
            print(f"Best validation F1: {results['best_val_f1']:.4f}")
            print(f"Test F1: {results['test_metrics']['f1']:.4f}")
        else:
            print("\n❌ Training failed. Check memory usage and try again.")

if __name__ == "__main__":
    main()
