"""
Simple Logger for Training (No TensorBoard Dependencies)
"""
import os
import json
from datetime import datetime

class SimpleLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.metrics = {}
        
    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({"step": step, "value": value})
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {tag}: {value} (step {step})\n")
    
    def close(self):
        """Save all metrics to JSON file"""
        metrics_file = self.log_file.replace('.log', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
