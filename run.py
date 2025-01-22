import os
import torch
import warnings
import numpy as np
import argparse
from model import HyReaL
from data_loader import load_data
from config_model import CompleteConfig
import torch.nn.functional as F
from experiment_logger import ExperimentLogger
from typing import List

warnings.filterwarnings('ignore')

def get_activation_function(name: str) -> callable:
    return getattr(F, name)

def build_layer_config(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> List[int]:
    """Builds a list of layer dimensions for the model"""
    if n_layers < 2:
        raise ValueError("Number of layers must be at least 2")
        
    layers = [input_dim]
    for _ in range(n_layers - 1):
        layers.append(hidden_dim)
    layers.append(output_dim)
    return layers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='cora',
                      choices=['acm', 'wiki', 'citeseer', 'dblp', 'film', 
                              'cornell', 'cora', 'wisc', 'uat', 'amap'],
                      help='Dataset name')
    parser.add_argument('-n', '--n_layers', type=int, default=2,
                      choices=[2, 4, 6, 8, 10],
                      help='Number of layers')
    args = parser.parse_args()
    
    # Initialize experiment logger with layer info
    logger = ExperimentLogger(f"oversmoothing_ablation_with_different_quaternion_layers_2_to_10", 
                              output_path="ablation_results/oversmoothing/")
    
    # Load config using dataclass
    config = CompleteConfig.from_yaml('config.yaml')
    
    # Set common parameters
    seed = config.common.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.common.gpu

    # Use dataset name from command line args
    name = args.dataset
    dataset_config = getattr(config, name)
    
    # Load data
    features, adjacency, labels = load_data(name)
    
    # Build layer config
    input_dim = dataset_config.layers[0]  # first layer dimension
    hidden_dim = dataset_config.layers[1] # middle layer dimension
    output_dim = dataset_config.layers[2] # last layer dimension
    layers = build_layer_config(input_dim, hidden_dim, output_dim, args.n_layers)
    acts = [get_activation_function(dataset_config.acts[0])] * len(layers)
    
    # Training setup
    acc_list, nmi_list, ari_list, f1_list = [], [], [], []

    for run_idx in range(config.common.runs):
        model = HyReaL(
            name, features, adjacency, labels,
            layers=layers,  
            acts=acts,
            max_epoch=dataset_config.max_epoch,
            max_iter=dataset_config.max_iter,
            coeff_reg=dataset_config.coeff_reg,
            learning_rate=dataset_config.learning_rate,
            seed=seed,
            lam=np.power(2.0, dataset_config.lamSC)
        )
        if torch.cuda.is_available():
            model.cuda() 
        else:
            model.cpu()
            
        model.pretrain(dataset_config.pre_iter, learning_rate=dataset_config.pretrain_learning_rate)
        acc, nmi, ari, f1 = model.run()

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

        # Record results for this run
        run_metrics = {
            "acc": float(acc),
            "nmi": float(nmi),
            "ari": float(ari),
            "f1": float(f1)
        }
        logger.add_run_result(
            name, 
            layers, 
            run_metrics,
            max_epoch=dataset_config.max_epoch,
            repeater_runs=config.common.runs
        )
        model = None
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)
        
    # Save all results
    logger.save_results()

    print("\n")
    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)

    print(acc_list.mean(), "±", acc_list.std())
    print(nmi_list.mean(), "±", nmi_list.std())
    print(ari_list.mean(), "±", ari_list.std())
    print(f1_list.mean(), "±", f1_list.std())


