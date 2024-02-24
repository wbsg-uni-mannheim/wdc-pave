import gzip
import json
import os
from pieutils import convert_tuple_keys_to_string, detect_data_types, save_populated_task_to_json
from pieutils.evaluation import evaluate_predictions_with_data_types, visualize_performance_by_data_types, evaluate_normalization_performance
from pieutils.preprocessing import load_known_attribute_values,load_known_attribute_values_for_normalized_attributes


def error_analysis(path_task_dict):
    with open(path_task_dict, 'r') as f:
        task_dict = json.load(f)

    if not os.path.exists('../prompts/runs/data_types'):
        os.makedirs('../prompts/runs/data_types')
    
    if not os.path.exists('../figures/data_types'):
        os.makedirs('../figures/data_types')

    task_dict = detect_data_types(task_dict)
    #save_populated_task_to_json(task_dict['task_name'], task_dict, title=True, description=True)

    results_by_attribute_data_type = evaluate_predictions_with_data_types(task_dict)
    print(results_by_attribute_data_type)
    visualize_performance_by_data_types(results_by_attribute_data_type, task_dict['task_name'], task_dict['dataset_name'], task_dict['timestamp'])
        
    converted_results = convert_tuple_keys_to_string(results_by_attribute_data_type)

    #print(converted_results)

    path_to_result_file = '../prompts/runs/data_types/{}__{}.json'.format(task_dict["task_name"], task_dict["dataset_name"])
    with open(path_to_result_file, 'w', encoding='utf-8') as fp:
        json.dump(converted_results, fp, indent=4, ensure_ascii=False)



if "__main__" == __name__:
    # Best Few-Shot
    path_task_dict = ""
    from_normalized = False
    error_analysis(path_task_dict)
