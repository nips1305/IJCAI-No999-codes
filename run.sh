#!/bin/bash

# log
mkdir -p logs

# acm wiki citeseer dblp film cornell cora wisc uat amap
datasets="acm wiki citeseer dblp film cornell cora wisc uat amap"
layers="2"

# iteration
for dataset in $datasets; do
    for n_layers in $layers; do
        echo "==========================================="
        echo "Dataset: $dataset, Layers: $n_layers"
        echo "==========================================="
        
        # run
        python run.py --dataset "$dataset" --n_layers "$n_layers" 2>&1 | tee "logs/${dataset}_${n_layers}layers.log"
        
        echo "Completed: $dataset with $n_layers layers"
        echo
    done
done

echo "All experiments completed!"
