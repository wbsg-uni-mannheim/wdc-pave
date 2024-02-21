import gzip
import json
import os

from pieutils import convert_tuple_keys_to_string, detect_data_types_for_normalized_data, save_populated_task_to_json
from pieutils.evaluation import evaluate_predictions_with_data_types, visualize_performance_by_data_types, evaluate_normalization_performance, evaluate_normalization_performance_one_by_one
from pieutils.preprocessing import load_known_attribute_values,load_known_attribute_values_for_normalized_attributes


def error_analysis(path_task_dict):
    with open(path_task_dict, 'r') as f:
        task_dict = json.load(f)
    
    if not os.path.exists('../prompts/runs/data_types'):
        os.makedirs('../prompts/runs/data_types')
    
    if not os.path.exists('../figures/data_types'):
        os.makedirs('../figures/data_types')

    if not os.path.exists('../figures/normalization '):
        os.makedirs('../figures/normalization ')

    task_dict = detect_data_types_for_normalized_data(task_dict, title=True, description=True, dataset="wdc")

    results_by_attribute_data_type = evaluate_predictions_with_data_types(task_dict)

    visualize_performance_by_data_types(results_by_attribute_data_type, task_dict['task_name'], task_dict['dataset_name'], task_dict['timestamp'])
            
    converted_results = convert_tuple_keys_to_string(results_by_attribute_data_type)

    # Converting tuple keys to string keys
    path_to_result_file = '../prompts/runs/data_types/{}__{}.json'.format(task_dict["task_name"], task_dict["dataset_name"])
    with open(path_to_result_file, 'w', encoding='utf-8') as fp:
        json.dump(converted_results, fp, indent=4, ensure_ascii=False)
        
    #evaluate_normalization_performance(task_dict['task_name'], converted_results)

    evaluate_normalization_performance_one_by_one(task_dict)

if "__main__" == __name__:
    # Few Shot best results
    # GPT 3.5:
    #path_task_dict= "../prompts/runs/task_run_chat_few_shot_normalization_only_10_shots_0.2_train_percentage_wdc_gpt-3.5-turbo-0613_2024-02-19_18-34-33_title_and_description.json"
    error_analysis(path_task_dict)
