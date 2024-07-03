import json
import os

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

PROCESSED_DATASETS = 'data/processed_datasets/old_structure/'


def calculate_dataset_statistics(loaded_dataset, category, descriptions):
    # Filter samples by category if provided
    filtered_dataset = [sample for sample in loaded_dataset if sample['category'] == category]
    filtered_descriptions = descriptions[descriptions['Category'] == category]

    # Number of attribute_value pairs
    no_attributes = len(filtered_descriptions)
    no_product_offers = len(filtered_dataset)
    no_attribute_value_pairs = 0
    no_na_values = 0
    unique_values = set()
    hostnames = set()
    for sample in filtered_dataset:
        hostnames.add(sample['url'])
        for index, attribute in filtered_descriptions.iterrows():
            no_attribute_value_pairs += 1
            for key in sample['target_scores'][attribute['Attribute']].keys():
                if key != 'n/a':
                    unique_values.add(f'{key}_{sample["target_scores"][attribute["Attribute"]][key]}')
                else:
                    no_na_values += 1

    print(f'Number of attributes: {no_attributes}')
    print(f'Number of attribute-value pairs: {no_attribute_value_pairs}')
    print(f'Number of n/a values: {no_na_values}')
    print(f'Number of unique values: {len(unique_values)}')
    print(f'Number of product offers: {no_product_offers}')
    print(f'Number of unique hostnames: {len(hostnames)}')

    # Put statistics into dictionary and return
    statistics = {
        'no_attributes': no_attributes,
        'no_product_offers': no_product_offers,
        'no_attribute_value_pairs': no_attribute_value_pairs,
        'no_na_values': no_na_values,  # 'n/a' values are not counted as unique values
        'no_unique_values': len(unique_values),
        'no_unique_hostnames': len(hostnames)
    }

    return statistics


@click.command
@click.option('--dataset', default='wdc', help='Dataset name')
def main(dataset):
    print(f'Dataset name: {dataset}')

    # Load descriptions
    descriptions = pd.read_csv(f'data/descriptions/{dataset}/descriptions.csv', sep=";")
    # Filter for attributes that require normalization
    descriptions = descriptions[pd.notna(descriptions['Normalization_params'])]

    # # Load dataset
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}/'

    loaded_datasets = {'train': [], 'test': []}
    with open(os.path.join(directory_path_preprocessed, 'train_0.2.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train'].append(json.loads(line))

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['test'].append(json.loads(line))

    # Load dataset - normalized
    # directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}/normalized'
    #
    # loaded_datasets = {'train': [], 'test': []}
    # with open(os.path.join(directory_path_preprocessed, 'normalized_train_0.2.jsonl'), 'r') as f:
    #     for line in tqdm(f.readlines()):
    #         loaded_datasets['train'].append(json.loads(line))
    #
    # with open(os.path.join(directory_path_preprocessed, 'normalized_test.jsonl'), 'r') as f:
    #     for line in tqdm(f.readlines()):
    #         loaded_datasets['test'].append(json.loads(line))


    print(f'Loaded {len(loaded_datasets["train"])} training samples and {len(loaded_datasets["test"])} testing samples')

    # Create a complete dataset
    complete_dataset = loaded_datasets['train'] + loaded_datasets['test']
    print(f'Complete dataset has {len(complete_dataset)} samples')

    # Number of unique categories
    categories = set()
    for sample in complete_dataset:
        categories.add(sample['category'])

    print(f'Number of unique categories: {len(categories)}')

    statistics_per_category = {}
    for category in categories:
        # Collect statistics
        print(f'Category: {category}')
        statistics_per_category[category] = calculate_dataset_statistics(complete_dataset, category, descriptions)
        print('')

    # Sum statistics over all categories
    print('Complete dataset statistics:')
    no_attributes = sum([statistics['no_attributes'] for statistics in statistics_per_category.values()])
    no_product_offers = sum([statistics['no_product_offers'] for statistics in statistics_per_category.values()])
    no_attribute_value_pairs = sum(
        [statistics['no_attribute_value_pairs'] for statistics in statistics_per_category.values()])
    no_na_values = sum([statistics['no_na_values'] for statistics in statistics_per_category.values()])
    no_unique_values = sum([statistics['no_unique_values'] for statistics in statistics_per_category.values()])
    no_unique_hostnames = sum([statistics['no_unique_hostnames'] for statistics in statistics_per_category.values()])

    print(f'Number of attributes: {no_attributes}')
    print(f'Number of attribute-value pairs: {no_attribute_value_pairs}')
    print(f'Number of n/a values: {no_na_values}')
    print(f'Number of unique values: {no_unique_values}')
    print(f'Number of product offers: {no_product_offers}')
    print(f'Number of unique hostnames: {no_unique_hostnames}')


    # Statistics for training set
    print('##################')
    print('Training set statistics:')
    print('##################')
    statistics_per_category = {}
    for category in categories:
        # Collect statistics
        print(f'Category: {category}')
        statistics_per_category[category] = calculate_dataset_statistics(loaded_datasets['train'], category, descriptions)
        print('')

    # Sum statistics over all categories
    print('Complete dataset statistics:')
    no_attributes = sum([statistics['no_attributes'] for statistics in statistics_per_category.values()])
    no_product_offers = sum([statistics['no_product_offers'] for statistics in statistics_per_category.values()])
    no_attribute_value_pairs = sum(
        [statistics['no_attribute_value_pairs'] for statistics in statistics_per_category.values()])
    no_unique_values = sum([statistics['no_unique_values'] for statistics in statistics_per_category.values()])
    no_unique_hostnames = sum([statistics['no_unique_hostnames'] for statistics in statistics_per_category.values()])

    print(f'Number of attributes: {no_attributes}')
    print(f'Number of attribute-value pairs: {no_attribute_value_pairs}')
    print(f'Number of unique values: {no_unique_values}')
    print(f'Number of product offers: {no_product_offers}')
    print(f'Number of unique hostnames: {no_unique_hostnames}')


    print('##################')
    print('Test set statistics:')
    print('##################')
    # Statistics for test set
    statistics_per_category = {}
    for category in categories:
        # Collect statistics
        print(f'Category: {category}')
        statistics_per_category[category] = calculate_dataset_statistics(loaded_datasets['test'], category, descriptions)
        print('')

    # Sum statistics over all categories
    print('Complete dataset statistics:')
    no_attributes = sum([statistics['no_attributes'] for statistics in statistics_per_category.values()])
    no_product_offers = sum([statistics['no_product_offers'] for statistics in statistics_per_category.values()])
    no_attribute_value_pairs = sum(
        [statistics['no_attribute_value_pairs'] for statistics in statistics_per_category.values()])
    no_unique_values = sum([statistics['no_unique_values'] for statistics in statistics_per_category.values()])
    no_unique_hostnames = sum([statistics['no_unique_hostnames'] for statistics in statistics_per_category.values()])

    print(f'Number of attributes: {no_attributes}')
    print(f'Number of attribute-value pairs: {no_attribute_value_pairs}')
    print(f'Number of unique values: {no_unique_values}')
    print(f'Number of product offers: {no_product_offers}')
    print(f'Number of unique hostnames: {no_unique_hostnames}')


if __name__ == '__main__':
    main()
