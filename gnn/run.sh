#!/bin/bash
#SBATCH -o result.txt  

model_types=("gin" "gcn" "gat" "graphsage")  

echo "Training started at $(date)"

for model_type in "${model_types[@]}"
do
    echo model type: $model_type
    
    python train.py  --gnn_type $model_type 
done

echo "Training completed at $(date)"
