# Organize datasets

# Target structure:
#   processed_datasets/
#       wdc/ # WDC PAVE dataset containing only attributes
#               that require normalization but not normalization of values is performed.
#           train.jsonl
#           train_large.jsonl # Contains additional training data for WDC dataset
#           test.jsonl
#
#       wdc_normalized/ # WDC PAVE dataset containing only attributes
#                           that require normalization and normalization of values is performed.
#           train.jsonl
#           train_large.jsonl # Contains additional training data for WDC dataset
#           test.jsonl
#
#       wdc_with_all_attributes/ # WDC PAVE dataset containing all attributes.
#           train.jsonl
#           train_large.jsonl # Contains additional training data for WDC dataset
#           test.jsonl
#
#       wdc_with_all_attributes_normalized/ # WDC PAVE dataset containing all attributes
#                                               and normalization of values is performed.
#           train.jsonl
#           train_large.jsonl # Contains additional training data for WDC dataset
#           test.jsonl

import os
import json
import pandas as pd
from tqdm import tqdm

DEPRECATED_DATASETS = 'data/processed_datasets/old_structure'
PROCESSED_DATASETS = 'data/processed_datasets'

def filter_attributes(sample, descriptions):
    category = sample['category']
    filtered_descriptions_for_sample = descriptions[descriptions['Category'] == category]
    filtered_target_scores = {}
    for attribute in filtered_descriptions_for_sample['Attribute'].unique():
        filtered_target_scores[attribute] = sample['target_scores'][attribute]

    sample['target_scores'] = filtered_target_scores
    return sample

def main():
    dataset = 'wdc'
    # Load descriptions
    descriptions = pd.read_csv(f'data/descriptions/{dataset}/descriptions.csv', sep=";")
    # Filter for attributes that require normalization
    descriptions = descriptions[pd.notna(descriptions['Normalization_params'])]

    # Load dataset
    directory_path_preprocessed = f'{DEPRECATED_DATASETS}/{dataset}/'

    loaded_datasets = {'train': [], 'train_large': [], 'test': [],
                       'train_normalized': [], 'train_normalized_large': [], 'test_normalized': []}
    with open(os.path.join(directory_path_preprocessed, 'train_0.2.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'train_1.0.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train_large'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['test'].append(json.loads(line))

    # Load dataset - normalized
    directory_path_preprocessed = f'{DEPRECATED_DATASETS}/{dataset}/normalized'

    with open(os.path.join(directory_path_preprocessed, 'normalized_train_0.2.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train_normalized'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'normalized_train_1.0.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train_normalized_large'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'normalized_test.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['test_normalized'].append(json.loads(line))


    print(f'Loaded {len(loaded_datasets["train"])} training samples, {len(loaded_datasets["train_large"])} training samples and {len(loaded_datasets["test"])} testing samples')
    print(f'Loaded {len(loaded_datasets["train_normalized"])} normalized training samples, {len(loaded_datasets["train_large"])} normalized training samples and {len(loaded_datasets["test_normalized"])} normalized testing samples')

    # Save dataset with all attributes
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}_with_all_attributes'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for sample in loaded_datasets['train']:
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'train_large.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_large']:
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for sample in loaded_datasets['test']:
            f.write(json.dumps(sample) + '\n')

    # Save normalized dataset with all attributes
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}_with_all_attributes_normalized'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_normalized']:
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'train_large.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_normalized_large']:
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for sample in loaded_datasets['test_normalized']:
            f.write(json.dumps(sample) + '\n')

    # Save dataset with only attributes that require normalization
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for sample in loaded_datasets['train']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'train_large.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_large']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for sample in loaded_datasets['test']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    # Save normalized dataset with only attributes that require normalization
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}_normalized'
    if not os.path.exists(directory_path_preprocessed):
        os.makedirs(directory_path_preprocessed)

    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_normalized']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'train_large.jsonl'), 'w') as f:
        for sample in loaded_datasets['train_normalized_large']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'w') as f:
        for sample in loaded_datasets['test_normalized']:
            # Filter for attributes that require normalization
            sample = filter_attributes(sample, descriptions)
            f.write(json.dumps(sample) + '\n')

    print('Datasets saved')


if __name__ == '__main__':
    main()
