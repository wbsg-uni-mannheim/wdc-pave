import json
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
import colorsys
import random 
import os
import pandas as pd

# Constants for evaluation dictionary keys
from pieutils import combine_example

NN = 'nn'
NV = 'nv'
VN = 'vn'
VC = 'vc'
VW = 'vw'


def calculate_evaluation_metrics_multiple_attributes(targets, preds, categories, unique_category,
                                                     unique_attribute):
    """Calculate evaluation metrics (nn, nv, vn, vc, vw) for a specific attribute-category combination in multi attribute setting.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.
        unique_category: The unique category to evaluate.
        unique_attribute: The unique attribute to evaluate.

    Returns:
        dict: A dictionary containing the counts of each evaluation metric.
    """
    eval_dict = {
        NN: 0,
        NV: 0,
        VN: 0,
        VC: 0,
        VW: 0
    }

    for target, pred, category in zip(targets, preds, categories):
        if unique_category != category or unique_attribute not in target:
            # Evaluate per attribute/category
            continue

        target_values = [value.strip() if value not in ["n/a", None] else None for value in target[unique_attribute]]
        try:
            #processed_attribute_name = unique_attribute.lower().replace(' ', '_')
            # print(processed_attribute_name)
            prediction = json.loads(pred)[unique_attribute] if unique_attribute in json.loads(
                pred) else None
            #print(prediction)
            prediction = prediction if prediction != "n/a" and prediction != "None" else None
        except:
            print('Not able to decode prediction: \n {}'.format(pred))
            prediction = None

        if target_values[0] is None and prediction is None:
            eval_dict[NN] += 1
        elif target_values[0] is None and prediction is not None:
            eval_dict[NV] += 1
        elif target_values[0] is not None and prediction is None:
            eval_dict[VN] += 1
        elif prediction in target_values:
            eval_dict[VC] += 1
        else:
            eval_dict[VW] += 1

    return eval_dict


def calculate_recall_precision_f1_multiple_attributes(targets, preds, categories, known_attributes):
    """Calculate recall, precision, and f1 for the extractions per category."""
    unique_categories = list(set(categories))

    result_dict = defaultdict(dict)
    total_eval = defaultdict(int)

    for unique_category in unique_categories:
        category_eval_dict = defaultdict(int)

        for unique_attribute in known_attributes[unique_category]:
            eval_dict = calculate_evaluation_metrics_multiple_attributes(
                targets, preds, categories, unique_category, unique_attribute)

            if sum(eval_dict.values()) == 0:
                # If there are no values to evaluate, skip this attribute-category combination.
                continue

            precision, recall, f1 = calculate_precision_recall_f1(eval_dict)

            total_eval = update_total_evaluation_metrics(total_eval, eval_dict)

            result_dict[f'{unique_attribute}__{unique_category}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            for key in eval_dict:
                category_eval_dict[key] += eval_dict[key]

        category_precision, category_recall, category_f1 = calculate_precision_recall_f1(category_eval_dict)
        
        result_dict[f'Category_{unique_category}'] = {
            'precision': category_precision,
            'recall': category_recall,
            'f1': category_f1, 
            'NN': category_eval_dict['nn'],
            'NV': category_eval_dict['nv'],
            'VN': category_eval_dict['vn'],
            'VC': category_eval_dict['vc'],
            'VW': category_eval_dict['vw']
        }

    macro_precision, macro_recall, macro_f1 = calculate_macro_scores(result_dict)
    micro_precision, micro_recall, micro_f1 = calculate_micro_scores(total_eval)

    result_dict['overall_counts'] = {
        'NN': total_eval['nn'],
        'NV': total_eval['nv'],
        'VN': total_eval['vn'],
        'VC': total_eval['vc'],
        'VW': total_eval['vw']
    }

    result_dict['macro'] = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    result_dict['micro'] = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return result_dict


def calculate_recall_precision_f1(targets, preds, categories, attributes):
    """Calculate recall, precision and f1 for the extractions.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.

    Returns:
        dict: A dictionary containing recall, precision, and f1 scores for each attribute-category combination,
              as well as macro and micro scores.
    """
    result_dict = defaultdict(dict)
    total_eval = defaultdict(int)

    for unique_category, unique_attribute in product(set(categories), set(attributes)):
        eval_dict = calculate_evaluation_metrics(targets, preds, categories, attributes, unique_category, unique_attribute)

        if eval_dict[NV] + eval_dict[VC] + eval_dict[VW] + eval_dict[VN] == 0:
            # If there are no values to evaluate, skip this attribute-category combination.
            continue
        precision, recall, f1 = calculate_precision_recall_f1(eval_dict)

        total_eval = update_total_evaluation_metrics(total_eval, eval_dict)

        result_dict[f'{unique_attribute}_{unique_category}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    macro_precision, macro_recall, macro_f1 = calculate_macro_scores(result_dict)
    micro_precision, micro_recall, micro_f1 = calculate_micro_scores(total_eval)

    result_dict['macro'] = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    result_dict['micro'] = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return result_dict

def calculate_evaluation_metrics(targets, preds, categories, attributes, unique_category, unique_attribute):
    """Calculate evaluation metrics (nn, nv, vn, vc, vw) for a specific attribute-category combination.

    Args:
        targets (list): List of target values.
        preds (list): List of predicted values.
        categories (list): List of categories.
        attributes (list): List of attributes.
        unique_category: The unique category to evaluate.
        unique_attribute: The unique attribute to evaluate.

    Returns:
        dict: A dictionary containing the counts of each evaluation metric.
    """
    eval_dict = {
        NN: 0,
        NV: 0,
        VN: 0,
        VC: 0,
        VW: 0
    }

    for target, pred, category, attribute in zip(targets, preds, categories, attributes):
        if unique_attribute != attribute or unique_category != category:
            continue

        target_values = [value.strip() if value != "n/a" else None for value in target]
        prediction = pred if pred != "n/a" else None

        if target_values[0] is None and prediction is None:
            eval_dict[NN] += 1
        elif target_values[0] is None and prediction is not None:
            eval_dict[NV] += 1
        elif target_values[0] is not None and prediction is None:
            eval_dict[VN] += 1
        elif prediction in target_values:
            eval_dict[VC] += 1
        else:
            eval_dict[VW] += 1

    return eval_dict

def calculate_precision_recall_f1(eval_dict):
    """Calculate precision, recall, and f1 scores based on the evaluation metrics.

    Args:
        eval_dict (dict): A dictionary containing the counts of each evaluation metric.

    Returns:
        tuple: A tuple containing precision, recall, and f1 scores.
    """
    precision = round((eval_dict[VC] / (eval_dict[NV] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[NV] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    recall = round((eval_dict[VC] / (eval_dict[VN] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[VN] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0

    return precision, recall, f1

def update_total_evaluation_metrics(total_eval, eval_dict):
    """Update the total counts of evaluation metrics.

    Args:
        total_eval (dict): A dictionary containing the total counts of each evaluation metric.
        eval_dict (dict): A dictionary containing the counts of each evaluation metric.

    Returns:
        dict: The updated total_eval dictionary.
    """
    for metric in eval_dict:
        total_eval[metric] += eval_dict[metric]
    return total_eval

def calculate_macro_scores(result_dict):
    """Calculate macro scores (precision, recall, and f1) based on the result dictionary.

    Args:
        result_dict (dict): The result dictionary containing attribute-category specific evaluation metrics.
        attributes (list): List of attributes.
        categories (list): List of categories.

    Returns:
        tuple: A tuple containing macro precision, recall, and f1 scores.
    """
    precision_scores = [result['precision'] for result in result_dict.values()]
    macro_precision = round(sum(precision_scores) / len(precision_scores), 2) if precision_scores else 0

    recall_scores = [result['recall'] for result in result_dict.values()]
    macro_recall = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0

    f1_scores = [result['f1'] for result in result_dict.values()]
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 2) if f1_scores else 0

    return macro_precision, macro_recall, macro_f1

def calculate_micro_scores(total_eval):
    """Calculate micro scores (precision, recall, and f1) based on the total evaluation counts.

    Args:
        total_eval (dict): A dictionary containing the total counts of each evaluation metric.

    Returns:
        tuple: A tuple containing micro precision, recall, and f1 scores.
    """
    micro_precision = round((total_eval[VC] / (total_eval[NV] + total_eval[VC] + total_eval[VW])) * 100, 2) if (total_eval[NV] + total_eval[VC] + total_eval[VW]) > 0 else 0
    micro_recall = round((total_eval[VC] / (total_eval[VN] + total_eval[VC] + total_eval[VW])) * 100, 2) if (total_eval[VN] + total_eval[VC] + total_eval[VW]) > 0 else 0
    micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 2) if (micro_precision + micro_recall) > 0 else 0

    return micro_precision, micro_recall, micro_f1


def evaluate_predictions(preds, task_dict, multiple_attribute=True):
    """Evaluate the task dictionary based on the predictions and return the evaluation metrics."""
    if multiple_attribute:
        targets = [example['target_scores'] for example in task_dict['examples']]

        categories = [example['category'] for example in task_dict['examples']]

        postprocessed_preds = [pred.json() if pred is not None else '' for pred in preds]

        task_dict['examples'] = [combine_example(example, pred, post_pred)
                                 for example, pred, post_pred in
                                 zip(task_dict['examples'], postprocessed_preds, postprocessed_preds)]

        results = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds,
                                                                                 categories,
                                                                                 task_dict['known_attributes'])
    else:
        targets = [example['target_scores'] for example in task_dict['examples']]

        categories = [example['category'] for example in task_dict['examples']]
        attributes = [example['attribute'] for example in task_dict['examples']]

        # postprocessed_preds = [pred.replace('\n','').replace('Answer: ', '').replace('AI: ', '').strip() for pred in preds]
        postprocessed_preds = [pred.split(':')[-1].split('\n')[0].strip() for pred, attribute in zip(preds, attributes)]
        postprocessed_preds = ['' if pred is None else pred for pred in postprocessed_preds]

        task_dict['examples'] = [combine_example(example, pred, post_pred)
                                 for example, pred, post_pred in zip(task_dict['examples'], preds, postprocessed_preds)]

        results = calculate_recall_precision_f1(targets, postprocessed_preds, categories, attributes)

    print(f'Task: {task_dict["task_name"]} on dataset: {task_dict["dataset_name"]}')
    print(results['micro'])
    print(f"{results['micro']['micro_precision']}\t{results['micro']['micro_recall']}\t{results['micro']['micro_f1']}")

    return results

def evaluate_predictions_from_file(task_dict, multiple_attribute=True):
    """Evaluate the task dictionary based on the predictions and return the evaluation metrics."""
    preds = [example['pred'] for example in task_dict['examples']]
    post_preds = [example['post_pred'] for example in task_dict['examples']]
    targets = [example['target_scores'] for example in task_dict['examples']]
    categories = [example['category'] for example in task_dict['examples']]

    if multiple_attribute:
        results = calculate_recall_precision_f1_multiple_attributes(targets, post_preds, categories, task_dict['known_attributes'])
    else:
        attributes = [attr for category in task_dict['known_attributes'].values() for attr in category]
        results = calculate_recall_precision_f1(targets, post_preds, categories, attributes)

    print(f'Task: {task_dict["task_name"]} on dataset: {task_dict["dataset_name"]}')
    print(results['micro'])
    print(f"{results['micro']['micro_precision']}\t{results['micro']['micro_recall']}\t{results['micro']['micro_f1']}")

    return results

def calculate_cost_per_1k(total_costs, task_dict):
    targets = [example['target_scores'] for example in task_dict['examples']]
    n_attributes = sum(len(attribute_dict) for attribute_dict in targets)

    return (total_costs/n_attributes) * 1000

def visualize_performance(task_dict):
    if not os.path.exists('../figures'):
        os.makedirs('../figures')

    dataset_name = task_dict["dataset_name"]
    task_name = task_dict['task_name']
    timestamp = task_dict["timestamp"]
    model = task_dict['model'].replace('/', '_')
    overall_f1 = task_dict["results"]["micro"]["micro_f1"]

    category_results = {}

    for row, scores in task_dict["results"].items():
        if row.startswith("Category"):
            category_name = row.split("Category_")[1]
            category_results[category_name] = scores['f1']

    sorted_categories = sorted(category_results.items(), key=lambda x: x[1])

    labels = [category for category, _ in sorted_categories]
    scores = [score for _, score in sorted_categories]  
    
    differences = [score - overall_f1 for score in scores]

    colors = ['red' if diff < 0 else 'green' for diff in differences]

    y = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    bars = ax.barh(y, differences, color=colors, left=overall_f1)  

    ax.set_ylabel('Categories')
    ax.set_xlabel('F1 Scores')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(x=overall_f1, color='black') 

    fig.tight_layout()
    plt.savefig(f'../figures/{dataset_name}_{task_name}_{model}_{timestamp}.svg')
    plt.close()

def visualize_performance_from_file(file_name):
    with open(f'../prompts/runs/{file_name}', 'r') as f:
        task_dict = json.load(f)

    dataset_name = task_dict["dataset_name"]
    task_name = task_dict['task_name']
    timestamp = task_dict["timestamp"]
    model = task_dict['model']
    overall_f1 = task_dict["results"]["micro"]["micro_f1"]

    category_results = {}

    for row, scores in task_dict["results"].items():
        if row.startswith("Category"):
            category_name = row.split("Category_")[1]
            category_results[category_name] = scores['f1']

    sorted_categories = sorted(category_results.items(), key=lambda x: x[1])

    labels = [category for category, _ in sorted_categories]
    scores = [score for _, score in sorted_categories]  
    differences = [score - overall_f1 for score in scores]
    colors = ['red' if diff < 0 else 'green' for diff in differences]

    y = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    bars = ax.barh(y, differences, color=colors, left=overall_f1)  
    ax.set_ylabel('Categories')
    ax.set_xlabel('F1 Scores')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(x=overall_f1, color='black') 

    fig.tight_layout()
    plt.savefig(f'../figures/{dataset_name}_{task_name}_{model}_{timestamp}.svg')
    plt.close()


def evaluate_predictions_with_data_types(task_dict):
    # Initialize dictionaries for counts
    overall_eval_dict = {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0}
    data_type_eval_dict = defaultdict(lambda: {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0})
    cat_attr_dtype_eval_dict = defaultdict(lambda: {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0, 'total': 0})

    for example in task_dict['examples']:
        preds = example['post_pred']
        if preds:
            predictions = json.loads(example['post_pred'])

            for attribute, target_values in example['target_scores'].items():
                predicted_value = predictions.get(attribute, None)
                predicted_value = None if predicted_value in ['n/a', None] else predicted_value
                category = example['category']

                actual_value_exists = any(value != 'n/a' for value in target_values)
                match_found = False

                for value, data_type in target_values.items():
                    if value != 'n/a' and predicted_value == value:
                        match_found = True
                        # Update counts using the data type of the matched value
                        update_counts('VC', overall_eval_dict, data_type_eval_dict, cat_attr_dtype_eval_dict, data_type, category, attribute)
                        break  # Correct prediction found, no need to check other values

                if not match_found:
                    if predicted_value is None and not actual_value_exists:
                        # NN Scenario
                        update_counts('NN', overall_eval_dict, data_type_eval_dict, cat_attr_dtype_eval_dict, None, category, attribute)
                    elif predicted_value is None and actual_value_exists:
                        # VN Scenario
                        update_counts_for_all_data_types('VN', overall_eval_dict, data_type_eval_dict, cat_attr_dtype_eval_dict, target_values, category, attribute)
                    elif predicted_value is not None and not actual_value_exists:
                        # NV Scenario
                        update_counts('NV', overall_eval_dict, data_type_eval_dict, cat_attr_dtype_eval_dict, None, category, attribute)
                    else:
                        # VW Scenario
                        update_counts_for_all_data_types('VW', overall_eval_dict, data_type_eval_dict, cat_attr_dtype_eval_dict, target_values, category, attribute)

    def calculate_metrics(eval_dict):
        precision = eval_dict['VC'] / (eval_dict['NV'] + eval_dict['VC'] + eval_dict['VW']) if eval_dict['NV'] + eval_dict['VC'] + eval_dict['VW'] else 0
        recall = eval_dict['VC'] / (eval_dict['VN'] + eval_dict['VC'] + eval_dict['VW']) if eval_dict['VN'] + eval_dict['VC'] + eval_dict['VW'] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return precision, recall, f1

    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1 = calculate_metrics(overall_eval_dict)

    # Calculate metrics by data type
    data_type_metrics = {data_type: calculate_metrics(counts) for data_type, counts in data_type_eval_dict.items()}

    # Calculate metrics by category-attribute-datatype
    cat_attr_dtype_metrics = {
    key: {
        'metrics': calculate_metrics(counts),
        'total': counts['total']  # Include total count
    } 
    for key, counts in cat_attr_dtype_eval_dict.items()
    }

    return {
    'overall': {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1, 'counts': overall_eval_dict},
    'by_data_type': data_type_metrics,
    'by_category_attribute_datatype': cat_attr_dtype_metrics
}

def update_counts(scenario, overall_dict, dtype_dict, cat_attr_dtype_dict, data_type, category, attribute):
    overall_dict[scenario] += 1
    if data_type:
        dtype_dict[data_type][scenario] += 1
        cat_attr_dtype_dict[(category, attribute, data_type)][scenario] += 1
        cat_attr_dtype_dict[(category, attribute, data_type)]['total'] += 1

def update_counts_for_all_data_types(scenario, overall_dict, dtype_dict, cat_attr_dtype_dict, target_values, category, attribute):
    overall_dict[scenario] += 1
    for value, data_type in target_values.items():
        if data_type != 'n/a':
            dtype_dict[data_type][scenario] += 1
            cat_attr_dtype_dict[(category, attribute, data_type)][scenario] += 1
            cat_attr_dtype_dict[(category, attribute, data_type)]['total'] += 1

def visualize_performance_by_data_types(results, task_name, dataset_name, timestamp):
    category_data = {}
    max_data_types = 0  # Track the maximum number of data types in any category

    # First pass to find the maximum number of data types
    for (category, attribute, datatype), data in results['by_category_attribute_datatype'].items():
        f1_score = data['metrics'][2]  # Access the F1 score
        if category not in category_data:
            category_data[category] = {}
        if attribute not in category_data[category]:
            category_data[category][attribute] = []
        category_data[category][attribute].append((datatype, f1_score))
        max_data_types = max(max_data_types, len(category_data[category][attribute]))

    # Define pastel colors
    pastel_red = (1, 0.6, 0.6)  # Lighter red
    pastel_green = (0.5, 0.8, 0.5)  # Muted green

    n_rows = min(5, len(category_data))
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * max_data_types))

    # If there's only one category, axes won't be an array, so wrap it in a list
    if n_rows == 1:
        axes = [axes]

    for idx, (category, attributes) in enumerate(category_data.items()):
        if idx >= 5: 
            break

        ax = axes[idx]
        flattened_data = [(attribute, datatype, score) for attribute, data in attributes.items() for datatype, score in data]
        flattened_data.sort(key=lambda x: x[2], reverse=True)  
        category_mean_f1 = np.mean([score for attr, dtype, score in flattened_data])

        labels = [f'{attr} ({dtype})' for attr, dtype, _ in flattened_data]
        scores = [score for _, _, score in flattened_data]
        colors = [pastel_red if score < category_mean_f1 else pastel_green for score in scores]
        y_pos = np.arange(len(labels))

        bar_height = 0.8 
        ax.barh(y_pos, scores, color=colors, edgecolor='black', height=bar_height)
        ax.set_ylim(-0.5, len(labels) - 0.5 + bar_height) 

        # Set y-ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, ha='right', fontsize=8)
        ax.set_xlabel('F1 Score')
        category_mean_f1 = np.mean([score for attr, dtype, score in flattened_data])
        ax.axvline(x=category_mean_f1, color='red', linewidth=1, linestyle='--')
        ax.set_title(f'{category} (Category Mean F1: {category_mean_f1:.2f})')

    plt.tight_layout(pad=0.5)
    plt.savefig(f'../figures/data_types/{dataset_name}_{task_name}_{timestamp}_combined.svg')
    plt.close()


    data_type_performance = results['by_data_type']

    # Sort and extract F1 scores
    sorted_data_types = sorted(data_type_performance.items(), key=lambda x: x[1][2], reverse=True)
    data_types, sorted_data = zip(*sorted_data_types)

    sorted_scores = [score[2] for score in sorted_data] 

    # Calculate the overall mean F1 score
    mean_f1_score = np.mean(sorted_scores)
    colors = [pastel_red if score < mean_f1_score else pastel_green for score in sorted_scores]

    if len(data_types) != len(sorted_scores):
        raise ValueError("Mismatch in the length of data types and scores.")

    # Create the plot for overall data type performance
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(data_types))
    bars = ax.barh(y_pos, sorted_scores, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data_types)
    ax.set_xlabel('F1 Score')
    ax.axvline(x=mean_f1_score, color='gray', linestyle='--', linewidth=1)
    ax.set_title('Performance by Data Type')

    plt.tight_layout()
    plt.savefig(f'../figures/data_types/{dataset_name}_{task_name}_{timestamp}_overall.svg')
    plt.close()

    
def evaluate_normalization_performance(task_name, converted_results):
    # Load normalization types from CSV
    directory_path = f"../data/descriptions/wdc/descriptions.csv"
    descriptions_csv = pd.read_csv(directory_path, sep=";")
    descriptions_csv["Normalization_params"] = descriptions_csv["Normalization_params"].str.strip("[]").str.replace("'", "")
    descriptions_csv["Normalization_params_general"] = descriptions_csv["Normalization_params_general"].str.strip("[]").str.replace("'", "")

    performance_data = []
    for key, values in converted_results["by_category_attribute_datatype"].items():
        category, attribute, data_type = key.split("__")
        metrics = values['metrics'] 
        total_examples = values['total'] 
        performance_data.append({
            "Category": category,
            "Attribute": attribute,
            "Data Type": data_type,
            "F1": metrics[2],
            "Precision": metrics[0],
            "Recall": metrics[1],
            "Total": total_examples
        })
    
    performance_df = pd.DataFrame(performance_data)
    merged_df = pd.merge(performance_df, descriptions_csv, on=["Category", "Attribute"])

    # Calculate weighted metrics
    weighted_metrics = merged_df.groupby("Normalization_params_general").apply(lambda x: 
        pd.Series({
            "F1_mean": np.average(x['F1'], weights=x['Total']),
            "F1_std": np.sqrt(np.cov(x['F1'], aweights=x['Total'])),
            "Precision_mean": np.average(x['Precision'], weights=x['Total']),
            "Precision_std": np.sqrt(np.cov(x['Precision'], aweights=x['Total'])),
            "Recall_mean": np.average(x['Recall'], weights=x['Total']),
            "Recall_std": np.sqrt(np.cov(x['Recall'], aweights=x['Total'])),
        })
    ).reset_index()

    adjusted_f1_stds = []
    adjusted_p_stds = []
    adjusted_r_stds = []
    for mean, std in zip(weighted_metrics['F1_mean'], weighted_metrics['F1_std']):
        max_possible_std = 1 - mean  
        adjusted_std = min(std, max_possible_std)  # Adjust std if it exceeds the max possible value
        adjusted_f1_stds.append(adjusted_std)
    for mean, std in zip(weighted_metrics['Precision_mean'], weighted_metrics['Precision_std']):
        max_possible_std = 1 - mean  
        adjusted_std = min(std, max_possible_std)  # Adjust std if it exceeds the max possible value
        adjusted_p_stds.append(adjusted_std)
    for mean, std in zip(weighted_metrics['Recall_mean'], weighted_metrics['Recall_std']):
        max_possible_std = 1 - mean 
        adjusted_std = min(std, max_possible_std)  # Adjust std if it exceeds the max possible value
        adjusted_r_stds.append(adjusted_std)

    weighted_metrics['Adjusted_F1_std'] = adjusted_f1_stds
    weighted_metrics['Adjusted_P_std'] = adjusted_p_stds
    weighted_metrics['Adjusted_R_std'] = adjusted_r_stds

    print(weighted_metrics)

    # Prepare data for plotting
    f1_means = weighted_metrics['F1_mean']
    f1_stds = weighted_metrics['F1_std']
    precision_means = weighted_metrics['Precision_mean']
    precision_stds = weighted_metrics['Precision_std']
    recall_means = weighted_metrics['Recall_mean']
    recall_stds = weighted_metrics['Recall_std']

    weighted_metrics.sort_values(by='F1_mean', ascending=False, inplace=True)

    height = 0.25

    ind = np.arange(len(weighted_metrics))

    fig, ax = plt.subplots(figsize=(10, 8))

    f1_bars = ax.barh([i - height for i in ind], weighted_metrics['F1_mean'], height, xerr=weighted_metrics['Adjusted_F1_std'], label='F1', color='0.6', capsize=5)
    precision_bars = ax.barh(ind, weighted_metrics['Precision_mean'], height, xerr=weighted_metrics['Adjusted_P_std'], label='Precision', color='0.4', capsize=5)
    recall_bars = ax.barh([i + height for i in ind], weighted_metrics['Recall_mean'], height, xerr=weighted_metrics['Adjusted_R_std'], label='Recall', color='0.2', capsize=5)
    ax.set_ylabel('Normalization Operations')
    ax.set_xlabel('Scores')
    ax.set_title('Model Performance by Normalization Operation')
    ax.set_yticks(ind)
    ax.set_yticklabels(weighted_metrics['Normalization_params_general'].values)  # Ensure this line matches your DataFrame structure

    ax.legend()

    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'../figures/normalization/{task_name}_performance_normalization.svg')
    plt.close()

    # Separate by Normalization Operation:
    data = []
    for key, values in converted_results["by_category_attribute_datatype"].items():
        category, attribute, data_type = key.split("__")
        metrics = values['metrics']  
        data.append({
            "Category": category,
            "Attribute": attribute,
            "Data Type": data_type,
            "Precision": metrics[0],  # Access Precision from metrics
            "Recall": metrics[1],     # Access Recall from metrics
            "F1": metrics[2]          # Access F1 from metrics
        })
    results_df = pd.DataFrame(data)
    results_df = pd.merge(results_df, descriptions_csv, on=["Category", "Attribute"])

    results_df = results_df.sort_values(by=['Normalization_params_general', 'F1'], ascending=[True, False])
    
    #print(results_df)
        
    unique_operations = results_df['Normalization_params_general'].unique()
    n_operations = len(unique_operations)

    fig, axes = plt.subplots(nrows=n_operations, figsize=(10, 5 * n_operations))

    if n_operations == 1:
        axes = [axes]

    for ax, operation in zip(axes, unique_operations):
        filtered_df = results_df[results_df['Normalization_params_general'] == operation].reset_index(drop=True)
        
        # Sort and prepare custom labels
        sorted_df = filtered_df.sort_values(by=['Category', 'Attribute', 'Data Type'])
        y_labels = []
        last_category_attribute = None
        for _, row in sorted_df.iterrows():
            category_attribute = f"{row['Category']} \n  {row['Attribute']}"
            #if category_attribute == last_category_attribute:
            #    # If same category and attribute as the last row, only add data type
            #    y_labels.append(row['Data Type'])
            #else:
            y_labels.append(f"{category_attribute} \n    {row['Data Type']}")
            #last_category_attribute = category_attribute

        ax.barh(range(len(sorted_df)), sorted_df['Precision'], height=0.2, color='0.6', label='Precision')
        ax.barh([i + 0.2 for i in range(len(sorted_df))], sorted_df['Recall'], height=0.2, color='0.4', label='Recall')
        ax.barh([i + 0.4 for i in range(len(sorted_df))], sorted_df['F1'], height=0.2, color='0.2', label='F1')

        ax.set_xlabel('Scores')
        ax.set_title(f'Performance for {operation}')
        ax.set_yticks([i + 0.2 for i in range(len(sorted_df))])
        ax.set_yticklabels(y_labels, fontsize=8)

        if ax is axes[0]:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f'../figures/normalization/{task_name}_performance_normalization_by_operation.svg')
    plt.close()

import copy

def evaluate_normalization_performance_one_by_one(original_task_dict):
    directory_path = f"../data/descriptions/wdc/descriptions.csv"
    descriptions_csv = pd.read_csv(directory_path, sep=";")
    descriptions_csv["Normalization_params"] = descriptions_csv["Normalization_params"].str.strip("[]").str.replace("'", "")
    descriptions_csv["Normalization_params_general"] = descriptions_csv["Normalization_params_general"].str.strip("[]").str.replace("'", "")

    # Filter out rows where Normalization_instruction is not null
    descriptions_csv = descriptions_csv[descriptions_csv['Normalization_params_general'].notnull()]

    # Get unique normalization parameters
    unique_normalization_params = descriptions_csv['Normalization_params_general'].unique()

    for normalization_param in unique_normalization_params:
        # Make a deep copy of the task_dict for each normalization parameter
        task_dict = copy.deepcopy(original_task_dict)

        # Filter attributes based on the current normalization parameter
        filtered_attributes = descriptions_csv[descriptions_csv['Normalization_params_general'] == normalization_param]

        category_attributes_with_guideline = {}
        for index, row in filtered_attributes.iterrows():
            category = row['Category']
            attribute = row['Attribute']
            if category not in category_attributes_with_guideline:
                category_attributes_with_guideline[category] = []
            if attribute not in category_attributes_with_guideline[category]:
                category_attributes_with_guideline[category].append(attribute)

        # Filter the task_dict based on the current normalization parameter
        for category, attributes in task_dict['known_attributes'].items():
            if category in category_attributes_with_guideline:
                task_dict['known_attributes'][category] = [attr for attr in attributes if attr in category_attributes_with_guideline[category]]
            else:
                task_dict['known_attributes'][category] = []

        # Assuming you have a function to filter examples based on valid attributes
        # This part of the code should remain similar to your original logic, adjusting targets and predictions accordingly

        targets = [example['target_scores'] for example in task_dict['examples']]
        preds = [json.loads(pred) if pred else {} for pred in [example['pred'] for example in task_dict['examples']]]
        categories = [example['category'] for example in task_dict['examples']]
        
        # Adjust predictions to only include valid attributes for the current normalization parameter
        postprocessed_preds = [json.dumps({attr: value for attr, value in pred.items() if attr in category_attributes_with_guideline.get(example['category'], [])}) for pred, example in zip(preds, task_dict['examples'])]

        # Update examples with filtered predictions
        task_dict['examples'] = [combine_example(example, pred, post_pred)
                                for example, pred, post_pred in zip(task_dict['examples'], postprocessed_preds, postprocessed_preds)]

        # Calculate and print performance metrics for the current normalization parameter
        results = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds, categories, task_dict['known_attributes'])
        print(f"Evaluating performance for normalization parameter: {normalization_param}")
        print(results['micro'])  # Adjust according to your metrics structure


