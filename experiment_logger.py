import json
from pathlib import Path
from typing import Dict, List
import numpy as np

class ExperimentLogger:
    def __init__(self, experiment_name: str, output_path: str = "results"):
        self.experiment_name = experiment_name
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.output_file = self.output_path / f"{self.experiment_name}_results.json"
        
        self.results = {
            "experiment_name": self.experiment_name,
            "datasets": {}
        }
        
        self.current_dataset = None
        self.current_metrics = []
        
    def add_run_result(self, dataset: str, layer_config: List[int], 
                      run_metrics: Dict[str, float], max_epoch: int,
                      repeater_runs: int):
        if self.current_dataset is None:
            self.current_dataset = dataset
            
        if dataset != self.current_dataset:
            raise ValueError(f"Cannot mix results from different datasets: {self.current_dataset} vs {dataset}")
            
        self.current_metrics.append({
            "layer_config": layer_config,
            "layer numbers": len(layer_config)-1,
            "max_epoch": max_epoch,
            "repeater_runs": repeater_runs,
            "metrics": run_metrics
        })
    
    def compute_statistics(self, metrics_list):
        values = {
            "acc": [],
            "nmi": [],
            "ari": [],
            "f1": []
        }
        
        for run in metrics_list:
            for key in values:
                values[key].append(run["metrics"][key])
                
        return {
            key: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
            for key, vals in values.items()
        }
    
    def save_results(self):
        if not self.current_metrics:
            return
            
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                self.results = json.load(f)
        
        layer_str = "_".join(map(str, self.current_metrics[0]["layer_config"]))
        if self.current_dataset not in self.results["datasets"].keys() or \
            any(layer_str == m for m in self.results["datasets"].keys()):
            self.results["datasets"][self.current_dataset] = {
                layer_str: {
                    "layer_config": self.current_metrics[0]["layer_config"],
                    "layer numbers": self.current_metrics[0]["layer numbers"],
                    "max_epoch": self.current_metrics[0]["max_epoch"],
                    "repeater_runs": self.current_metrics[0]["repeater_runs"],
                    "runs": [m["metrics"] for m in self.current_metrics],
                    "mean_metrics": self.compute_statistics(self.current_metrics)
                }
            }
        else:
            self.results["datasets"][self.current_dataset].update({
                layer_str: {
                    "layer_config": self.current_metrics[0]["layer_config"],
                    "layer numbers": self.current_metrics[0]["layer numbers"],
                    "max_epoch": self.current_metrics[0]["max_epoch"],
                    "repeater_runs": self.current_metrics[0]["repeater_runs"],
                    "runs": [m["metrics"] for m in self.current_metrics],
                    "mean_metrics": self.compute_statistics(self.current_metrics)
                }
            })
        
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results for dataset {self.current_dataset} saved to {self.output_file}")
        
        self.current_dataset = None
        self.current_metrics = []
