#!/bin/bash
export PYTHONPATH=$(pwd)

datasets=("wdc") 
model="gpt-3.5-turbo-0613" 
shots=(3 5 10)
train_percentage=( 0.2 ) 
example_selectors=( "SemanticSimilarity" ) 
schema_type="json_schema"
title=True
description=True
force_from_different_website=False
separate=False
replace_example_values=True


#for dataset in "${datasets[@]}"
#do
#    for shot in "${shots[@]}"
#    do
#      echo "Running experiments for $dataset with $shot shots"
#      for percentage in "${train_percentage[@]}"
#      do
#        for example_selector in "${example_selectors[@]}"
#        do
#          python3 ../prompts/5_in-context_learning/in_context_list.py --dataset $dataset --shots $shot --model $model --train_percentage $train_percentage --example_selector $example_selector --title $title --description $description --force_from_different_website $force_from_different_website --separate $separate
#        done
#      done
#    done
#done


no_example_values=10
for dataset in "${datasets[@]}"
  do
    for shot in "${shots[@]}"
      do
        echo "Running experiments for $dataset with $shot shots"
        for percentage in "${train_percentage[@]}"
          do
          for example_selector in "${example_selectors[@]}"
            do
              python prompts/5_in-context_learning/in_context_schema_description_with_example_values.py --dataset $dataset --shots $shot --model $model --train_percentage $percentage --example_selector $example_selector --with_containment False --with_validation_error_handling False --title $title --description $description --force_from_different_website $force_from_different_website --separate $separate --no_example_values $no_example_values --replace_example_values $replace_example_values
          done
        done
    done
done