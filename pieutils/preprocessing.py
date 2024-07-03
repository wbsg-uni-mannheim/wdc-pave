import copy
import json
import os
import random
import pandas as pd 
import click
from dotenv import load_dotenv
from tqdm import tqdm
import gzip

# new imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import plotly.express as px
import plotly.io as pio
from collections import defaultdict
from urllib.parse import urlparse

from pieutils.config import RAW_DATA_SET_SOURCES, PROCESSED_DATASETS, MAVE_PROCECCESSED_DATASETS, \
    OPENTAG_PROCECCESSED_DATASETS
#from config import RAW_DATA_SET_SOURCES, PROCESSED_DATASETS, MAVE_PROCECCESSED_DATASETS, \
#    OPENTAG_PROCECCESSED_DATASETS

def update_task_dict_from_test_set(task_dict, title, description):
    """Update task dict from test set."""
    directory_path = f'{PROCESSED_DATASETS}/{task_dict["dataset_name"]}/'
    file_path = os.path.join(directory_path, 'test.jsonl')
    task_dict['known_attributes'] = {}

    if task_dict["dataset_name"] == "oa-mine":
        with open(file_path, 'r') as f:
            for line in f.readlines():
                record = json.loads(line)
                task_dict['examples'].append(record)
                if record['category'] not in task_dict['known_attributes']:
                    task_dict['known_attributes'][record['category']] = []
                for attribute in record['target_scores']:
                    if attribute not in task_dict['known_attributes'][record['category']]:
                        task_dict['known_attributes'][record['category']].append(attribute)
        print(task_dict['known_attributes'])
        
    elif task_dict["dataset_name"] in ["mave_random", "wdc"]:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                record = json.loads(line)
                input_title = record.get("input_title", "")
                input_description = record.get("input_description", "")
                category = record.get("category", "")
                pairs = record.get("target_scores", {})
                url = record.get("url", "")
                pairs_to_keep = {}

                for attribute, value in pairs.items():
                    filtered_values = {}
                    for inner_key, inner_value in value.items():
                        # Check if inner_value is a dictionary and has 'pid'
                        if isinstance(inner_value, dict) and 'pid' in inner_value:
                            pid_list = inner_value.get('pid', [])
                            if title and description and any(pid in pid_list for pid in [0, 1]):
                                filtered_values[inner_key] = 1
                            elif title and not description and 0 in pid_list:
                                filtered_values[inner_key] = 1
                            elif description and not title and 1 in pid_list:
                                filtered_values[inner_key] = 1
                        elif inner_key == 'n/a':  # Always include 'n/a' attributes
                            filtered_values[inner_key] = 1
                    
                    if filtered_values:
                        pairs_to_keep[attribute] = filtered_values
                    
                has_any_non_na = any(
                    inner_key != 'n/a'
                    for inner_values in pairs_to_keep.values()
                    for inner_key in inner_values
                )

                if has_any_non_na:
                    new_record = {
                                "input_title": input_title,
                                "input_description": input_description,
                                "category": category,
                                "target_scores": pairs_to_keep,
                                "url": url
                            }

                    task_dict['examples'].append(new_record)
                    if category not in task_dict['known_attributes']:
                        task_dict['known_attributes'][category] = list(pairs_to_keep.keys())

    # Save the updated task_dict as a JSON file
    if not os.path.exists('../prompts/runs'):
        os.makedirs('../prompts/runs')

    with open('../prompts/runs/updated_task_dict.json', 'w', encoding='utf-8') as json_file:
        json.dump(task_dict, json_file, ensure_ascii=False, indent=4)

    return task_dict 

def update_task_dict_from_normalized_test_set(task_dict, file_name, title, description, normalized_only, normalized_attributes):
    """Update task dict from test set."""
    directory_path = f'{PROCESSED_DATASETS}/{task_dict["dataset_name"]}/normalized/'
    file_path = os.path.join(directory_path, f'{file_name}.jsonl')
    task_dict['known_attributes'] = {}

    if task_dict["dataset_name"] in ["wdc"]:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                record = json.loads(line)
                input_title = record.get("input_title", "")
                input_description = record.get("input_description", "")
                category = record.get("category", "")
                pairs = record.get("target_scores", {})
                url = record.get("url", "")
                pairs_to_keep = {}

                for attribute, value in pairs.items():
                    filtered_values = {}
                    for inner_key, inner_value in value.items():
                        # Check if inner_value is a dictionary and has 'pid'
                        if isinstance(inner_value, dict) and 'pid' in inner_value:
                            pid_list = inner_value.get('pid', [])
                            if title and description and any(pid in pid_list for pid in [0, 1]):
                                filtered_values[inner_key] = 1
                            elif title and not description and 0 in pid_list:
                                filtered_values[inner_key] = 1
                            elif description and not title and 1 in pid_list:
                                filtered_values[inner_key] = 1
                        elif inner_key == 'n/a':  # Always include 'n/a' attributes
                            filtered_values[inner_key] = 1
                    
                    if filtered_values:
                        pairs_to_keep[attribute] = filtered_values
                    
                has_any_non_na = any(
                    inner_key != 'n/a'
                    for inner_values in pairs_to_keep.values()
                    for inner_key in inner_values
                )

                if has_any_non_na:
                    new_record = {
                                "input_title": input_title,
                                "input_description": input_description,
                                "category": category,
                                "target_scores": pairs_to_keep,
                                "url": url
                            }

                    task_dict['examples'].append(new_record)
                    if category not in task_dict['known_attributes']:
                        task_dict['known_attributes'][category] = list(pairs_to_keep.keys())

        if normalized_only:
            # Remove categories not in normalized_attributes
            task_dict['examples'] = [example for example in task_dict['examples'] if example['category'] in normalized_attributes]
            task_dict['known_attributes'] = {category: attrs for category, attrs in task_dict['known_attributes'].items() if category in normalized_attributes}

            for example in task_dict['examples']:
                category = example['category']
                if category in normalized_attributes:
                    valid_attributes = normalized_attributes[category]
                    example['target_scores'] = {attr: val for attr, val in example['target_scores'].items() if attr in valid_attributes}

            for category, attributes in task_dict['known_attributes'].items():
                valid_attributes = normalized_attributes[category]
                task_dict['known_attributes'][category] = [attr for attr in attributes if attr in valid_attributes]


    # Save the updated task_dict as a JSON file
    if not os.path.exists('../prompts/runs'):
        os.makedirs('../prompts/runs')
        
    # Save the updated task_dict as a JSON file
    with open('../prompts/runs/updated_task_dict.json', 'w', encoding='utf-8') as json_file:
        json.dump(task_dict, json_file, ensure_ascii=False, indent=4)

    return task_dict

def load_known_attribute_values(dataset_name, title, description, n_examples=5, consider_casing=False, train_percentage=1.0, test_set=False):
    """Loads known attribute values from train set."""
    known_attribute_values = {}
    if consider_casing:
        known_attribute_values_casing = {}
    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if test_set:
        file_path = os.path.join(directory_path, f'test.jsonl')
    else:
        if train_percentage < 1.0:
            file_path = os.path.join(directory_path, f'train_{train_percentage}.jsonl')
        else:
            file_path = os.path.join(directory_path, 'train.jsonl')

    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            category = record['category']
            if category not in known_attribute_values:
                known_attribute_values[category] = {}
                if consider_casing:
                    known_attribute_values_casing[category] = {}

            if dataset_name in ["wdc", "mave_random"]:
                for attribute, values in record['target_scores'].items():
                    if attribute not in known_attribute_values[category]:
                        known_attribute_values[category][attribute] = []
                        if consider_casing:
                            known_attribute_values_casing[category][attribute] = []

                    for value, details in values.items():
                        if isinstance(details, dict) and 'pid' in details:
                            include_value = False
                            if title and description:
                                include_value = any(pid in details['pid'] for pid in [0, 1])
                            elif title:
                                include_value = 0 in details['pid']
                            elif description:
                                include_value = 1 in details['pid']

                            if include_value:
                                if value not in known_attribute_values[category][attribute] and value != 'n/a' \
                                        and len(known_attribute_values[category][attribute]) < n_examples:
                                    if consider_casing:
                                        value_lower = value.lower()
                                        if value_lower not in known_attribute_values_casing[category][attribute]:
                                            known_attribute_values_casing[category][attribute].append(value_lower)
                                    else:
                                        known_attribute_values[category][attribute].append(value)

            else:     
                for attribute in record['target_scores']:
                    if attribute not in known_attribute_values[category]:
                        known_attribute_values[category][attribute] = []
                        if consider_casing:
                            known_attribute_values_casing[category][attribute] = []
                    for value in record['target_scores'][attribute]:
                        if value not in known_attribute_values[category][attribute] and value != 'n/a' \
                                and len(known_attribute_values[category][attribute]) < n_examples:
                            if consider_casing:
                                if value.lower() not in known_attribute_values_casing[category][attribute]:
                                    known_attribute_values[category][attribute].append(value)
                                    known_attribute_values_casing[category][attribute].append(value.lower())
                            else:
                                known_attribute_values[category][attribute].append(value)
    return known_attribute_values

def load_known_attribute_values_for_normalized_attributes(dataset_name, title, description, normalized_only, normalized_attributes, file_name_train, n_examples=5, consider_casing=False, train_percentage=1.0, test_set=False):
    """Function reads known attribute values only for the attributes to be normalized"""
    known_attribute_values = {}
    if consider_casing:
        known_attribute_values_casing = {}
    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if test_set:
        file_path = os.path.join(directory_path, f'test.jsonl')
    else:
        if train_percentage <= 1.0:
            file_path = os.path.join(directory_path, f'train_{train_percentage}.jsonl')
        else:
            file_path = os.path.join(directory_path, 'train.jsonl')

    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            category = record['category']

            # Skip categories not in normalized_attributes when normalized_only is True
            if normalized_only and category not in normalized_attributes:
                continue

            if category not in known_attribute_values:
                known_attribute_values[category] = {}
                if consider_casing:
                    known_attribute_values_casing[category] = {}

            if dataset_name in ["wdc"]:
                for attribute, values in record['target_scores'].items():
                    # Skip attributes not in normalized_attributes when normalized_only is True
                    if normalized_only and attribute not in normalized_attributes[category]:
                        continue

                    if attribute not in known_attribute_values[category]:
                        known_attribute_values[category][attribute] = []
                        if consider_casing:
                            known_attribute_values_casing[category][attribute] = []

                    for value, details in values.items():
                        if isinstance(details, dict) and 'pid' in details:
                            include_value = False
                            if title and description:
                                include_value = any(pid in details['pid'] for pid in [0, 1])
                            elif title:
                                include_value = 0 in details['pid']
                            elif description:
                                include_value = 1 in details['pid']

                            if include_value:
                                if value not in known_attribute_values[category][attribute] and value != 'n/a' \
                                        and len(known_attribute_values[category][attribute]) < n_examples:
                                    if consider_casing:
                                        value_lower = value.lower()
                                        if value_lower not in known_attribute_values_casing[category][attribute]:
                                            known_attribute_values_casing[category][attribute].append(value_lower)
                                    else:
                                        known_attribute_values[category][attribute].append(value)

    return known_attribute_values

def load_dataset(file_path):
    """Load a dataset from a JSON lines file."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_known_value_correspondences_for_normalized_attributes(dataset_name, normalized_only, normalized_attributes, file_name_train, n_examples=5, consider_casing=False, train_percentage=1.0, test_set=False):
    """Function reads known attribute values only for the attributes to be normalized"""
    known_attribute_values = {}
    if consider_casing:
        known_attribute_values_casing = {}

    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
    file_path = os.path.join(directory_path, f'test.jsonl' if test_set else f'train_{train_percentage}.jsonl' if train_percentage < 1.0 else 'train.jsonl')

    # Load normalized train set
    directory_path_norm = f'{PROCESSED_DATASETS}/{dataset_name}/normalized/'
    file_path_normalized = os.path.join(directory_path_norm, f'normalized_train_{train_percentage}.jsonl' if train_percentage < 1.0 else 'normalized_train_1.0.jsonl')

    original_entries = load_dataset(file_path)
    normalized_entries = load_dataset(file_path_normalized)

    # Create a map for quick lookup of normalized entries by id
    normalized_entries_map = {entry['id']: entry for entry in normalized_entries}

    # Process each original entry to find correspondences
    for original_entry in original_entries:
        id = original_entry['id']
        normalized_entry = normalized_entries_map.get(id)
        if not normalized_entry:
            continue

        category = original_entry['category']
        if normalized_only and category not in normalized_attributes:
            continue

        if category not in known_attribute_values:
            known_attribute_values[category] = {}
            if consider_casing:
                known_attribute_values_casing[category] = {}

        for attribute in normalized_attributes.get(category, []):
            original_values = original_entry['target_scores'].get(attribute, {})
            normalized_values = normalized_entry['target_scores'].get(attribute, {})

            # Select the first value for this attribute from both original and normalized datasets
            original_value_list = list(filter(lambda x: x != 'n/a', original_values.keys()))
            normalized_value_list = list(filter(lambda x: x != 'n/a', normalized_values.keys()))

            if original_value_list and normalized_value_list:
                original_value = original_value_list[0]
                normalized_value = normalized_value_list[0]

                correspondence = f"{original_value} -> {normalized_value}"
                if len(known_attribute_values[category].get(attribute, [])) < n_examples:
                    if correspondence not in known_attribute_values[category].get(attribute, []):
                        known_attribute_values[category].setdefault(attribute, []).append(correspondence)
                        if consider_casing:
                            correspondence_lower = correspondence.lower()
                            known_attribute_values_casing[category].setdefault(attribute, []).append(correspondence_lower)

    return known_attribute_values if not consider_casing else known_attribute_values_casing

def load_known_attribute_values_from_normalized(dataset_name, title, description, normalized_only, normalized_attributes, file_name_train, n_examples=5, consider_casing=False, train_percentage=1.0, test_set=False):
    """Loads known attribute values from normalized (!) train set, considering normalized attributes if specified."""
    known_attribute_values = {}
    if consider_casing:
        known_attribute_values_casing = {}

    directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/normalized/'
    file_path = os.path.join(directory_path, f'test.jsonl' if test_set else f'{file_name_train}.jsonl')

    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            category = record['category']

            # Skip categories not in normalized_attributes when normalized_only is True
            if normalized_only and category not in normalized_attributes:
                continue

            if category not in known_attribute_values:
                known_attribute_values[category] = {}
                if consider_casing:
                    known_attribute_values_casing[category] = {}

            if dataset_name in ["wdc"]:
                for attribute, values in record['target_scores'].items():
                    # Skip attributes not in normalized_attributes when normalized_only is True
                    if normalized_only and attribute not in normalized_attributes[category]:
                        continue

                    if attribute not in known_attribute_values[category]:
                        known_attribute_values[category][attribute] = []
                        if consider_casing:
                            known_attribute_values_casing[category][attribute] = []

                    for value, details in values.items():
                        if isinstance(details, dict) and 'pid' in details:
                            include_value = False
                            if title and description:
                                include_value = any(pid in details['pid'] for pid in [0, 1])
                            elif title:
                                include_value = 0 in details['pid']
                            elif description:
                                include_value = 1 in details['pid']

                            if include_value:
                                if value not in known_attribute_values[category][attribute] and value != 'n/a' \
                                        and len(known_attribute_values[category][attribute]) < n_examples:
                                    if consider_casing:
                                        value_lower = value.lower()
                                        if value_lower not in known_attribute_values_casing[category][attribute]:
                                            known_attribute_values_casing[category][attribute].append(value_lower)
                                    else:
                                        known_attribute_values[category][attribute].append(value)

    return known_attribute_values


def save_train_test_splits(dataset_name, train_examples, test_examples):
    # Save train and test splits
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    # Save train split
    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')

    # Save test split
    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')

    print(f'Saved train and test splits for {dataset_name}.')


def stratified_split(data, test_size=0.25, random_state=None):
    random.seed(random_state)

    categories_dict = {}
    for item in data:
        category = item.get("category")
        if category not in categories_dict:
            categories_dict[category] = []
        categories_dict[category].append(item)

    train_data, test_data = [], []
    for category, items in categories_dict.items():
        random.shuffle(items)
        split_index = int(len(items) * test_size)
        test_data.extend(items[:split_index])
        train_data.extend(items[split_index:])

    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data

def extract_domain(url):
    domain = urlparse(url).netloc
    return domain

def preprocess_wdc(dataset_name='wdc'):
    """Preprocess WDC"""
    directory_path = RAW_DATA_SET_SOURCES[dataset_name]
    product_dict = []
    for filename in tqdm(sorted(os.listdir(directory_path))):
        if filename.endswith("jsonl"):
            with open(os.path.join(directory_path, filename)) as file:
                total_lines = sum(1 for line in file)
                file.seek(0) 
                for line in tqdm(file, total=total_lines, desc=f'Processing {filename}', unit='line'):
                    product_dict.append(json.loads(line))
    
    urls = pd.read_csv("../data/source_data/id_url_mapping.csv.gz", compression="gzip")
    urls["domain"] = urls["url"].apply(lambda x: extract_domain(x))
    urls = urls.drop(columns="url")
    id_to_url = urls.set_index('id')['domain'].to_dict()

    for item in product_dict:
        item_id = item['id']
        # Add the URL to the dictionary if the ID exists in the DataFrame
        if item_id in id_to_url:
            item['url'] = id_to_url[item_id]

    train_examples, test_examples = stratified_split(
        product_dict, test_size=0.25, random_state=42)

    # Save train split
    with open(os.path.join("../data/processed_datasets/old_structure/wdc/", "train.jsonl"), "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    # Save test split
    with open(os.path.join("../data/processed_datasets/old_structure/wdc/", "test.jsonl"), "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")

    for percentage in [0.2, 1.0]:
        reduce_training_set_size("wdc", percentage)

    sunburst_data_list = []
    for example in product_dict:
        category = example['category']
        attributes = example['target_scores']
        
        for attribute, values in attributes.items():
            for value, details in values.items():
                if value == 'n/a':
                    value_presence = 'Negative'
                else:
                    # Assuming 'details' is a dictionary with a 'score' key
                    score = details.get('score', 0)
                    value_presence = 'Positive' if score > 0 else 'Negative'
                
                sunburst_data_list.append({'Category': category, 'Attribute': attribute, 'Value_Presence': value_presence})

    sunburst_data = pd.DataFrame(sunburst_data_list)

    fig = px.sunburst(sunburst_data, path=['Category', 'Attribute', 'Value_Presence'])
    fig.update_layout(
        title_text="",
        font_size=10,
        margin=dict(t=0, l=0, r=0, b=0), 
        height=700,
        width=700
    )
    pio.write_image(fig, '../resources/WDC_sunburst_attributes.svg')

    categories_products = defaultdict(list)
    for example in product_dict:
        categories_products[example['category']].append(example)

    for category, products in categories_products.items():
        value_counts = defaultdict(int)

        for product in products:
            pairs = product['target_scores']
            for attribute, values in pairs.items():
                count = len(values) if values else 0
                value_counts[count] += 1 
        
        total_attributes = sum(value_counts.values())
        
        average_counts = {k: v/total_attributes for k, v in value_counts.items()}
        
        plt.bar(average_counts.keys(), average_counts.values())  
        plt.title(f'Average Value Counts for Category: {category}')
        plt.xlabel('Number of accepted values')
        plt.ylabel('Average Occurrence')
        plt.xticks(range(0, max(value_counts.keys())+1))
        plt.savefig(f'../resources/histogram_value_counts_{category[:10]}_WDC.svg')
        plt.close()


def convert_to_mave_dataset(dataset_name, percentage=1, skip_test=False, normalization_subset=True):
    # Load dataset
    # Save train and test splits
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'

    if dataset_name == 'mave':
        # Read train split
        train_examples = []
        if percentage == 1:
            with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example['input'])
        else:
            with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example['input'])

        # Read test split
        test_examples = []
        if not skip_test:
            with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    test_examples.append(example['input'])

        print('Number of train examples: {}'.format(len(train_examples)))
        print('Number of test examples: {}'.format(len(test_examples)))

        # Identify records in original splits
        print('Identifying records in original splits...')
        records = {'train': {'positives': [], 'negatives': []}, 'test': {'positives': [], 'negatives': []}}
        for split in ['test', 'train']:
            directory_path = f'{RAW_DATA_SET_SOURCES[dataset_name]}/{split}/00_All'
            for filename in sorted(list(os.listdir(directory_path)), reverse=True):
                if '.jsonl' in filename:
                    print('Processing file: {}'.format(filename))
                    record_type = filename.replace('.jsonl', '').replace('mave_', '')
                    with open(f'{directory_path}/{filename}', 'r') as f:
                        for line in tqdm(f.readlines()):
                            record = json.loads(line)
                            title = [paragraph['text'] for paragraph in record['paragraphs'] if paragraph['source'] == 'title'][0]
                            if title in train_examples or title in test_examples:
                                records[split][record_type].append(record)

    elif dataset_name in ["oa-mine", "AE-110K"]:
        # Read train split
        train_examples = []
        if percentage == 1:
            with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)
        else:
            with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)

        # Read test split
        test_examples = []
        if not skip_test:
            with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    test_examples.append(example)

        print('Number of train examples: {}'.format(len(train_examples)))
        print('Number of test examples: {}'.format(len(test_examples)))
        # Convert training and test records to mave format
        print('Converting training and test records to mave format...')
        records = {'train': {'positives': [], 'negatives': []}, 'test': {'positives': [], 'negatives': []}}

        # Record format: {'id': str, 'category': str, 'paragraphs': [{'text': str, 'source': str}],
        #                 'attributes': [{'key': str, 'evidences': ['value': str, 'pid': int, "begin": int, "end": int]}]}

        # Train records
        print('Converting train records...')
        current_id = 0
        for example in tqdm(train_examples):
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}], 'attributes': []}
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [{'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['train']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['train']['negatives'].append(specific_record)
                    current_id += 1

        for example in tqdm(test_examples):
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}],
                      'attributes': []}
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [
                        {'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['test']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['test']['negatives'].append(specific_record)
                    current_id += 1

    elif dataset_name in ["wdc", "mave_random"]:
        # Read train split
        train_examples = []
        if percentage == 1:
            with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)
        else:
            with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    train_examples.append(example)

        # Read test split
        test_examples = []
        if not skip_test:
            with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
                for line in f.readlines():
                    example = json.loads(line)
                    test_examples.append(example)

        if normalization_subset:
            directory_path = f"../data/descriptions/wdc/descriptions.csv"
            descriptions_csv = pd.read_csv(directory_path, sep=";")

            # Filter out rows where Normalization_instruction is not null
            descriptions_csv = descriptions_csv[descriptions_csv['Normalization_instruction'].notnull()]

            category_attributes_with_guideline = {}

            for index, row in descriptions_csv.iterrows():
                category = row['Category']
                attribute = row['Attribute']
                if category not in category_attributes_with_guideline:
                    category_attributes_with_guideline[category] = []
                if attribute not in category_attributes_with_guideline[category]:
                    category_attributes_with_guideline[category].append(attribute)

            #print(category_attributes_with_guideline)

                # Filter examples
            for example in train_examples:
                category = example['category']
                if category in category_attributes_with_guideline:
                    valid_attributes = category_attributes_with_guideline[category]
                    example['target_scores'] = {attr: scores for attr, scores in example['target_scores'].items() if attr in valid_attributes}
            for example in test_examples:
                category = example['category']
                if category in category_attributes_with_guideline:
                    valid_attributes = category_attributes_with_guideline[category]
                    example['target_scores'] = {attr: scores for attr, scores in example['target_scores'].items() if attr in valid_attributes}

        print('Number of train examples: {}'.format(len(train_examples)))
        print('Number of test examples: {}'.format(len(test_examples)))
        # Convert training and test records to mave format
        print('Converting training and test records to mave format...')
        records = {'train': {'positives': [], 'negatives': []}, 'test': {'positives': [], 'negatives': []}}

        # Record format: {'id': str, 'category': str, 'paragraphs': [{'text': str, 'source': str}],
        #                 'attributes': [{'key': str, 'evidences': ['value': str, 'pid': int, "begin": int, "end": int]}]}

        # Train records
        print('Converting train records...')
        current_id = 0
        for example in tqdm(train_examples):  
            example["input"] = f"{example['input_title']} \n {example['input_description']}"
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}], 'attributes': []}   
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [{'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['train']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['train']['negatives'].append(specific_record)
                    current_id += 1

        for example in tqdm(test_examples):
            example["input"] = f"{example['input_title']} \n {example['input_description']}"
            record = {'category': example['category'], 'paragraphs': [{'text': example['input'], 'source': 'title'}], 'attributes': []}
            for attribute, value in example['target_scores'].items():
                specific_record = copy.deepcopy(record)
                specific_record['id'] = str(current_id)

                target_value = list(value.keys())[0]
                if target_value != "n/a":
                    begin = example['input'].find(target_value)
                    end = begin + len(target_value)
                    specific_record['attributes'] = [{'key': attribute, 'evidences': [
                        {'value': target_value, 'pid': 0, 'begin': begin, 'end': end}]}]
                    records['test']['positives'].append(specific_record)
                    current_id += 1
                else:
                    specific_record['attributes'] = [{'key': attribute, 'evidences': []}]
                    records['test']['negatives'].append(specific_record)
                    current_id += 1

    # Split train records into train and validation
    print('Splitting train records into train and validation...')
    random.shuffle(records['train']['positives'])
    random.shuffle(records['train']['negatives'])
    records['validation'] = {'positives': records['train']['positives'][:int(len(records['train']['positives']) * 0.1)],
                             'negatives': records['train']['negatives'][:int(len(records['train']['negatives']) * 0.1)]}
    records['train']['positives'] = records['train']['positives'][int(len(records['train']['positives']) * 0.1):]
    records['train']['negatives'] = records['train']['negatives'][int(len(records['train']['negatives']) * 0.1):]

    
    train_positives_count, train_negatives_count = count_attribute_values(records, 'train')
    print(f"Number of attribute-values in train positives: {train_positives_count}")
    print(f"Number of attribute-values in train negatives: {train_negatives_count}")

    # Count for validation set
    validation_positives_count, validation_negatives_count = count_attribute_values(records, 'validation')
    print(f"Number of attribute-values in validation positives: {validation_positives_count}")
    print(f"Number of attribute-values in validation negatives: {validation_negatives_count}")


    # Save records
    print('Saving records...')
    for split in ['test', 'validation', 'train']:
        if skip_test and split == 'test':
            continue
        for record_type in ['positives', 'negatives']:
            if percentage == 1:
                directory_path_preprocessed_mave = f'{MAVE_PROCECCESSED_DATASETS}/splits/PRODUCT/{split}/{dataset_name}'
            else:
                directory_path_preprocessed_mave = f'{MAVE_PROCECCESSED_DATASETS}/splits/PRODUCT/{split}/{dataset_name}_{percentage}'
            if not os.path.exists(directory_path_preprocessed_mave):
                os.makedirs(directory_path_preprocessed_mave)

            with open(f'{directory_path_preprocessed_mave}/mave_{record_type}.jsonl', 'w') as f:
                file_content = [json.dumps(record) for record in records[split][record_type]]
                f.write('\n'.join(file_content))

def count_attribute_values(records, split):
    positive_attributes_count = sum(len(item['attributes']) for item in records[split]['positives'])
    negative_attributes_count = sum(len(item['attributes']) for item in records[split]['negatives'])
    return positive_attributes_count, negative_attributes_count

def convert_to_open_tag_format(dataset_name, percentage=1, skip_test=False, normalization_subset=True):
    """Convert records to OpenTag format.
        Format 1: {"id": 19, "title": "热风2019年春季新款潮流时尚男士休闲皮鞋透气低跟豆豆鞋h40m9107", "attribute": "款式", "value": "豆豆鞋"}
        Format 2: "热风2019年春季新款潮流时尚男士休闲皮鞋透气低跟豆豆鞋h40m9107<$$$>款式<$$$>豆豆鞋<$$$>19" - Split by <$$$>
    """
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    # Read train split
    train_examples = []
    if percentage == 1:
        with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                train_examples.append(example)
    else:
        with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                train_examples.append(example)

    # Read test split
    test_examples = []
    if not skip_test:
        with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
            for line in f.readlines():
                example = json.loads(line)
                test_examples.append(example)

    print('Number of train examples: {}'.format(len(train_examples)))
    print('Number of test examples: {}'.format(len(test_examples)))
    # Convert training and test records to mave format
    print('Converting training and test records to mave format...')

    converted_train_examples = []
    converted_test_examples = []

    # Convert train examples
    if dataset_name not in ["mave_random", "wdc"]:
        record_id = 0
        for example in tqdm(train_examples):
            for attribute, value in example['target_scores'].items():
                for target_value in example['target_scores'][attribute]:
                    for part in ['input', 'description', 'features', 'rest']:
                        if part in example and example[part] is not None:
                            if target_value != "n/a":
                                if target_value in example[part]:
                                    record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                    record['attribute'] = attribute
                                    record['value'] = target_value
                                    converted_train_examples.append(record)
                                    record_id += 1
                            else:
                                record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                record['attribute'] = attribute
                                record['value'] = None
                                converted_train_examples.append(record)
                                record_id += 1

        # Convert test examples
        for example in tqdm(test_examples):
            for attribute, value in example['target_scores'].items():
                for target_value in example['target_scores'][attribute]:
                    for part in ['input', 'description', 'features', 'rest']:
                        if part in example and example[part] is not None:
                            if target_value != "n/a":
                                if target_value in example[part]:
                                    record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                    record['attribute'] = attribute
                                    record['value'] = target_value
                                    converted_test_examples.append(record)
                                    record_id += 1
                            else:
                                record = {'id': record_id, 'title': example[part], 'attribute': '', 'value': '', 'category': example['category']}
                                record['attribute'] = attribute
                                record['value'] = None
                                converted_test_examples.append(record)
                                record_id += 1
    
    else:
        if normalization_subset:
            directory_path = f"../data/descriptions/wdc/descriptions.csv"
            descriptions_csv = pd.read_csv(directory_path, sep=";")

            # Filter out rows where Normalization_instruction is not null
            descriptions_csv = descriptions_csv[descriptions_csv['Normalization_instruction'].notnull()]

            category_attributes_with_guideline = {}

            for index, row in descriptions_csv.iterrows():
                category = row['Category']
                attribute = row['Attribute']
                if category not in category_attributes_with_guideline:
                    category_attributes_with_guideline[category] = []
                if attribute not in category_attributes_with_guideline[category]:
                    category_attributes_with_guideline[category].append(attribute)

                # Filter examples
            for example in train_examples:
                category = example['category']
                if category in category_attributes_with_guideline:
                    valid_attributes = category_attributes_with_guideline[category]
                    example['target_scores'] = {attr: scores for attr, scores in example['target_scores'].items() if attr in valid_attributes}
            for example in test_examples:
                category = example['category']
                if category in category_attributes_with_guideline:
                    valid_attributes = category_attributes_with_guideline[category]
                    example['target_scores'] = {attr: scores for attr, scores in example['target_scores'].items() if attr in valid_attributes}

        record_id = 0
        for example in tqdm(train_examples):
            for attribute, values in example['target_scores'].items():
                for target_value, info in values.items():
                    combined_title = example.get('input_title', '') + " " + example.get('input_description', '')
                    combined_title = combined_title.strip()  # Remove leading/trailing whitespace
                    if target_value != "n/a":
                        # Create record with combined title and description
                        record = {
                            'id': record_id,
                            'title': combined_title,
                            'attribute': attribute,
                            'value': target_value,
                            'category': example['category'],
                        }
                        converted_train_examples.append(record)
                        record_id += 1
                    else:
                        # Create record for "n/a" with combined title and description
                        record = {
                            'id': record_id,
                            'title': combined_title,
                            'attribute': attribute,
                            'value': None,  # No value for "n/a"
                            'category': example['category'],
                        }
                        converted_train_examples.append(record)
                        record_id += 1

        # Convert test examples
        for example in tqdm(test_examples):
            for attribute, values in example['target_scores'].items():
                for target_value, info in values.items():
                    combined_title = example.get('input_title', '') + " " + example.get('input_description', '')
                    combined_title = combined_title.strip()  # Remove leading/trailing whitespace
                    if target_value != "n/a":
                        # Create record with combined title and description
                        record = {
                            'id': record_id,
                            'title': combined_title,
                            'attribute': attribute,
                            'value': target_value,
                            'category': example['category'],
                        }
                        converted_test_examples.append(record)
                        record_id += 1
                    else:
                        # Create record for "n/a" with combined title and description
                        record = {
                            'id': record_id,
                            'title': combined_title,
                            'attribute': attribute,
                            'value': None,  # No value for "n/a"
                            'category': example['category'],
                        }
                        converted_test_examples.append(record)
                        record_id += 1

    print(f'Converting train examples of length {len(converted_train_examples)}...')
    print(f'Converting test examples of length {len(converted_test_examples)}...')

    # Convert examples to second format
    print('Converting examples to second format...')
    converted_examples_format_2 = []
    random.shuffle(converted_train_examples)
    split_converted_train_examples = int(len(converted_train_examples) * 0.9)
    converted_train_examples_format_2 = converted_train_examples[:split_converted_train_examples]
    converted_valid_examples_format_2 = converted_train_examples[split_converted_train_examples:]
    for example in tqdm(converted_train_examples_format_2):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>train'
        converted_examples_format_2.append(example_format_2)

    for example in tqdm(converted_valid_examples_format_2):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>valid'
        converted_examples_format_2.append(example_format_2)

    for example in tqdm(converted_test_examples):
        example_format_2 = f'{example["title"]}<$$$>{example["category"]}-{example["attribute"]}<$$$>{example["value"]}<$$$>test'
        converted_examples_format_2.append(example_format_2)

    # Save records
    print('Saving records...')
    for split in ['test', 'train']:
        if skip_test and split == 'test':
            continue

        directoryname = f'{dataset_name}_{split}'
        if percentage != 1 and split == 'train':
            directoryname = f'{dataset_name}_{split}_{str(percentage).replace(".", "_")}'

        directory_path_preprocessed_opentag = f'{OPENTAG_PROCECCESSED_DATASETS}/{dataset_name}/{directoryname}'
        if not os.path.exists(directory_path_preprocessed_opentag):
            os.makedirs(directory_path_preprocessed_opentag)

        if split == 'test':
            with open(f'{directory_path_preprocessed_opentag}/test_sample.json', 'w', encoding='utf-8') as f:
                file_content = [json.dumps(record) for record in converted_test_examples]
                f.write('\n'.join(file_content))

        else:
            with open(f'{directory_path_preprocessed_opentag}/train_sample.json', 'w', encoding='utf-8') as f:
                file_content = [json.dumps(record) for record in converted_train_examples]
                f.write('\n'.join(file_content))

    directory_path_preprocessed_opentag = f'{OPENTAG_PROCECCESSED_DATASETS}/{dataset_name}'
    if not os.path.exists(directory_path_preprocessed_opentag):
        os.makedirs(directory_path_preprocessed_opentag)
    # Format 2
    with open(f'{directory_path_preprocessed_opentag}/{dataset_name}_{str(percentage).replace(".", "_")}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_examples_format_2))


def reduce_training_set_size(dataset_name, percentage):
    """Reduces the size of the training set of a dataset to the specified percentage of the original size."""
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset_name}/'
    # Read train split
    train_examples = []
    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
        for line in f.readlines():
            example = json.loads(line)
            train_examples.append(example)

    # Stratified sampling per category
    categories = list(set([example['category'] for example in train_examples]))
    train_examples_reduced = []
    for category in categories:
        examples_category = [example for example in train_examples if example['category'] == category]
        train_examples_reduced += random.sample(examples_category, int(len(examples_category)*percentage))

    # Save reduced training set
    with open(os.path.join(directory_path_preprocessed, f'train_{percentage}.jsonl'), 'w') as f:
        for example in train_examples_reduced:
            f.write(json.dumps(example) + '\n')
