import gzip
import json

from pieutils import combine_example, subset_task_dict,save_populated_task_to_json
from pieutils.evaluation import calculate_recall_precision_f1_multiple_attributes
from pieutils.preprocessing import load_known_attribute_values,load_known_attribute_values_for_normalized_attributes


def error_analysis(path_task_dict, from_normalized):
    with open(path_task_dict, 'r') as f:
        task_dict = json.load(f)

    # Evaluate performance only for normalizable attributes
    task_dict_subset = subset_task_dict(task_dict)

    targets = [example['target_scores'] for example in task_dict_subset['examples']]
    preds = [example['pred'] for example in task_dict_subset['examples']]
    categories = [example['category'] for example in task_dict_subset['examples']]
    postprocessed_preds = [pred if pred is not None else '' for pred in preds]

    task_dict_subset['examples'] = [combine_example(example, pred, post_pred)
                            for example, pred, post_pred in
                            zip(task_dict_subset['examples'], postprocessed_preds, postprocessed_preds)]

    task_dict_subset['results'] = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds, categories,
                                                                            task_dict_subset['known_attributes'])


    print(task_dict_subset['results']['micro'])

    task_dict_subset['task_name'] = f"{task_dict['task_name']}_subset"

    save_populated_task_to_json(task_dict_subset['task_name'], task_dict, title=True, description=True)

if "__main__" == __name__:
    path_task_dict = "../prompts/runs/"
    from_normalized = False
    error_analysis(path_task_dict, from_normalized)
