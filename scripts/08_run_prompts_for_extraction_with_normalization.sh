#!/bin/bash

export PYTHONPATH=$(pwd)

verbose=True
datasets=("wdc") 
models=("gpt-3.5-turbo-0613")  
schema_type="json_schema" 
train_percentage=0.2
replace_example_values=True
title=True
description=True
example_values=(0 3 5 10)
n_shots=(3 5 10)
example_selector="SemanticSimilarity"
normalization_params="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Name Generalisation', 'Unit Expansion', 'To Uppercase', 'Delete Marks']"
normalized_only=True
force_from_different_website=False

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Zero Shot List
        #python3 prompts/08_extraction_with_normalization/9_zero_shot_single_normalization_list.py --dataset $dataset --model $model --title $title --description $description  --normalization_params "$normalization_params" --normalized_only $normalized_only

        for no_example_values in "${example_values[@]}"; do
            # Run Zero-Shot with Descriptions and Correspondences for each no_example_values
            python3 prompts/08_extraction_with_normalization/8_zero_shot_normalization_with_correspondence_examples.py --dataset $dataset --model $model --title $title --description $description --train_percentage $train_percentage  --normalization_params "$normalization_params" --schema_type $schema_type --normalized_only $normalized_only --no_example_values $no_example_values --verbose $verbose
        done
    done
done


#for dataset in "${datasets[@]}"; do
#    for model in "${models[@]}"; do
#        for shots in "${n_shots[@]}"; do
#            # In-Context Demonstrations 
#            python3 prompts/08_extraction_with_normalization/9_few_shot_single_normalization.py --dataset $dataset --model $model --title $title --description $description --train_percentage $train_percentage  --normalization_params "$normalization_params" --normalized_only $normalized_only --shots $shots --example_selector $example_selector --force_from_different_website $force_from_different_website --verbose $verbose
#        done
#    done
#done

n_example_values=10
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for shots in "${n_shots[@]}"; do
            for no_example_values in "${n_example_values[@]}"; do
            # In-Context Demonstrations with descriptions and example values
            python3 prompts/08_extraction_with_normalization/8_few_shot_normalization_description_with_example_correspondences.py --dataset $dataset --model $model --title $title --description $description --train_percentage $train_percentage  --normalization_params "$normalization_params" --normalized_only $normalized_only --shots $shots --example_selector $example_selector --force_from_different_website $force_from_different_website --verbose $verbose --no_example_values $no_example_values --replace_example_values $replace_example_values --schema_type $schema_type
            done
        done
    done
done