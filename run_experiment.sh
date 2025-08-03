#!/bin/bash

set -e

echo "Starting Sinhala Tokenization Experiment Suite"

DATASETS=(
    "train.csv"
    "augmented_datasets/dataset_minor_typos.csv"
    "augmented_datasets/dataset_aggressive_typos.csv"
    "augmented_datasets/dataset_mixed_coding.csv"
)

TOKENIZERS=("byte" "char" "word" "wpe" "sinlib") 

for dataset_path in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset_path" .csv | sed 's/dataset_//')

    for tokenizer_type in "${TOKENIZERS[@]}"; do
        RUN_NAME="${tokenizer_type}-${dataset_name}"
        
        echo "---------------------------------------------------------"
        echo "RUNNING: ${RUN_NAME}"
        echo "DATASET: ${dataset_path}"
        echo "TOKENIZER: ${tokenizer_type}"
        echo "---------------------------------------------------------"
        
        python train.py \
            --data_path "$dataset_path" \
            --tokenizer_type "$tokenizer_type" \
            --run_name "$RUN_NAME" \
            --num_epochs 1 \
            --batch_size 32 \
            --hidden_size 512 \
            --num_layers 4 \
            --num_heads 8 \
            --intermediate_size 512 \
            --dropout_prob 0.2 \
            --learning_rate 5e-5

        echo "FINISHED: ${RUN_NAME}"
    done
done

echo "Experiment suite complete."
