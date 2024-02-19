#!/bin/bash

export PYTHONPATH="/mnt/c/Users/nbaum/Dropbox/wdc-pave/:$PYTHONPATH"

verbose=True
datasets=("wdc") 
models=("gpt-3.5-turbo-0613")  # "gpt-4-1106-preview" "gpt-3.5-turbo-0613"
schema_type="json_schema" 
train_percentage=1.0
with_containment=False
replace_example_values=False
title=True
description=True
no_example_values=10 # correspondences in correspondence setting
shots=10
example_selector="SemanticSimilarity"
normalization_params="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Name Generalisation', 'Unit Expansion', 'To Uppercase', 'Delete Marks']"
normalized_only=True
hint=False
force_from_different_website=False


for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Zero Shot List
        #python3 ../prompts/08_normalization/8_zero_shot_normalization_list.py --dataset $dataset --model $model --title $title --description $description  --normalization_params "$normalization_params" --normalized_only $normalized_only 
        
        # Run Zero-Shot with Descriptions and Correspondences
        #python3 ../prompts/08_normalization/8_zero_shot_normalization_with_correspondence_examples.py --dataset $dataset --model $model --title $title --description $description --train_percentage $train_percentage  --normalization_params "$normalization_params" --schema_type $schema_type --normalized_only $normalized_only --no_example_values $no_example_values

        # In-Context Demonstrations 
        #python3 ../prompts/08_normalization/8_few_shot_normalization.py --dataset $dataset --model $model --title $title --description $description --train_percentage $train_percentage  --normalization_params "$normalization_params" --normalized_only $normalized_only --shots $shots --example_selector $example_selector --force_from_different_website $force_from_different_website --verbose $verbose
    done
done