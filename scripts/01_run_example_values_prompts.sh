#!/bin/bash
export PYTHONPATH="/mnt/c/Users/nbaum/Dropbox/wdc-pave/:$PYTHONPATH"

datasets=("wdc") 
models=("gpt-4-1106-preview") 
schema_types=("json_schema") 
example_values_counts=(3 5 10)  
train_percentage=0.2
with_containment=False
replace_example_values=False
title=True
description=True
separate=False

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python3 ../prompts/1_zero_shot_list/zero_shot_list.py --dataset $dataset --model $model --title $title --description $description --separate $separate
        
        for no_example_values in "${example_values_counts[@]}"; do  
            for schema_type in "${schema_types[@]}"; do
                echo "Running experiments for $dataset, $model, $schema_type with $no_example_values example values"
                python3 ../prompts/2_zero_shot_schema/zero_shot_schema_description_with_example_values.py --dataset $dataset --model $model --schema_type $schema_type --train_percentage $train_percentage --with_containment $with_containment --replace_example_values $replace_example_values --no_example_values $no_example_values --title $title --description $description --separate $separate
            done
        done
    done
done