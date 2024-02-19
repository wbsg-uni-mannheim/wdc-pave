#!/bin/bash
export PYTHONPATH="/mnt/c/Users/nbaum/Dropbox/wdc-pave/:$PYTHONPATH"

datasets=("wdc") 
model="gpt-4-1106-preview" # "gpt-4-1106-preview" "gpt-3.5-turbo-0613"
shots=(3 5 10)
train_percentage=( 0.2 ) 
example_selectors=( "SemanticSimilarity" ) 
schema_type="list"
title=True
description=True
force_from_different_website=False
separate=False


for dataset in "${datasets[@]}"
do
    for shot in "${shots[@]}"
    do
      echo "Running experiments for $dataset with $shot shots"
      for percentage in "${train_percentage[@]}"
      do
        for example_selector in "${example_selectors[@]}"
        do
          python3 ../prompts/5_in-context_learning/in_context_list.py --dataset $dataset --shots $shots --model $model --train_percentage $train_percentage --example_selector $example_selector --title $title --description $description --force_from_different_website $force_from_different_website --separate $separate
        done
      done
    done
done
