import json
import os
import random

import click
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



from pieutils.config import PROCESSED_DATASETS


def mostly_numeric(s):
    # Count the number of numeric characters
    numeric_count = sum(c.isdigit() for c in s)

    # Check if more than half of the characters are numeric
    return numeric_count > len(s) / 2


def is_all_numeric(s):
    return s.isdigit()


# Load dataset
@click.command
@click.option('--dataset', default='mave', help='Dataset name')
@click.option('--title', default=True, help='Calculate Statistics for title')
@click.option('--description', default=False, help='Calculate Statistics for description')
def main(dataset, title, description):
    print(f'Dataset name: {dataset}')

    # Load dataset
    directory_path_preprocessed = f'{PROCESSED_DATASETS}/{dataset}/'

    loaded_datasets = {'train': [], 'test': []}
    with open(os.path.join(directory_path_preprocessed, 'train.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['train'].append(json.loads(line))          

    with open(os.path.join(directory_path_preprocessed, 'test.jsonl'), 'r') as f:
        for line in tqdm(f.readlines()):
            loaded_datasets['test'].append(json.loads(line))

    has_non_na_overall = 0
    if dataset in ["mave_random", "wdc"]:
        for split, records in loaded_datasets.items():
            new_records = []
            has_non_na = 0 
            for record in records:
                input_title = record.get("input_title", "")
                input_description = record.get("input_description", "")
                category = record.get("category", "")
                pairs = record.get("target_scores", {})
                pairs_to_keep = {}
                url = record.get("url", None)
                
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
                    new_records.append(new_record)
                    has_non_na += 1
                    has_non_na_overall += 1
            
            loaded_datasets[split] = new_records
            print(f"Split {split} has {has_non_na} valid products with at least one positive pair")
            print(f"Dataset has {has_non_na_overall} valid products with at least one positive pair")

    # Calculate number of titles and attributes that have at least one on n/a attribute. If they dont, discard them
    unique_categories = set()
    unique_attributes = set()
    unique_category_attribute_combinations = set()
    attributes_by_category = {'train': {}, 'test': {}}
    records_by_category = {'train': {}, 'test': {}}
    no_attributes_by_product = {'train': [], 'test': []}
    no_negatives = {'train': 0, 'test': 0}
    attribute_values_by_catgory_attribute = {'train': {}, 'test': {}}
    token_attr_pair_data = {
        'title': defaultdict(lambda: {'tokens': 0, 'attr_pairs': 0}),
        'description': defaultdict(lambda: {'tokens': 0, 'attr_pairs': 0}),
        'concatination': defaultdict(lambda: {'tokens': 0, 'attr_pairs': 0})
        }

    print("---NUMBER OF TOKENS IN TITLE")

    # Calculate no. tokens
    encoding = tiktoken.get_encoding("cl100k_base")

    num_title_tokens_by_product = []
    num_description_tokens_by_product = []
    num_feature_tokens_by_products = []
    num_rest_tokens_by_products = []

    # Iterate over all records
    for split, records in loaded_datasets.items():
        for record in records:
            unique_categories.add(record['category'])
            if record['category'] not in records_by_category[split]:
                records_by_category[split][record['category']] = []
            records_by_category[split][record['category']].append(record)
            no_attributes_by_product[split].append(len(record['target_scores']))
            for attribute, attribute_values in record['target_scores'].items():
                unique_category_attribute_combinations.add(f'{attribute}-{record["category"]}')
                unique_attributes.add(f'{attribute}')

                if record['category'] not in attributes_by_category[split]:
                    attributes_by_category[split][record['category']] = {}
                if attribute not in attributes_by_category[split][record['category']]:
                    attributes_by_category[split][record['category']][attribute] = {'positive': 0, 'negative': 0 }

                #attributes_by_category[record['category']][attribute] += 1
                if 'n/a' in attribute_values.keys():
                    attributes_by_category[split][record['category']][attribute]['negative'] += 1
                else:
                    attributes_by_category[split][record['category']][attribute]['positive'] += 1

                if record['category'] not in attribute_values_by_catgory_attribute[split]:
                    attribute_values_by_catgory_attribute[split][record['category']] = {}

                if attribute not in attribute_values_by_catgory_attribute[split][record['category']]:
                    attribute_values_by_catgory_attribute[split][record['category']][attribute] = set()

                if 'n/a' in attribute_values.keys():
                    no_negatives[split] += 1

                for attribute_value in attribute_values.keys():
                    if attribute_value != 'n/a':
                        attribute_values_by_catgory_attribute[split][record['category']][attribute].add(attribute_value)

            # Calculate no. tokens
            if dataset == "mave_random" or dataset == "wdc":
                num_title_tokens_by_product.append(len(encoding.encode(record['input_title'])))
                num_description_tokens_by_product.append(len(encoding.encode(record['input_description'])))
            else:
                num_title_tokens_by_product.append(len(encoding.encode(record['input'])))
                #if 'title' in record:
                #    print(record)
                #    num_title_tokens_by_product.append(len(encoding.encode(record['title'])))
                #if 'description' in record:
                #    num_description_tokens_by_product.append(len(encoding.encode(record['description'])))
                #if 'features' in record:
                #    num_feature_tokens_by_products.append(len(encoding.encode(record['features'])))
                #if 'rest' in record:
                #    num_rest_tokens_by_products.append(len(encoding.encode(record['rest'])))

    avg_title_tokens = sum(num_title_tokens_by_product) / len(num_title_tokens_by_product) if num_title_tokens_by_product and title else 0 
    avg_description_tokens = sum(num_description_tokens_by_product) / len(num_description_tokens_by_product) if num_description_tokens_by_product and description else 0

    print(f"Avg. Number of tokens per title: {avg_title_tokens}")
    print(f"Avg. Number of tokens per description: {avg_description_tokens}")


    print("AVERAGE NUMBER OF PAIRS BY SPLIT")
    category_avg_data = defaultdict(lambda: defaultdict(lambda: {'pos_count': 0, 'neg_count': 0, 'record_count': 0}))
    for split, records in loaded_datasets.items():
        split_pos_count = 0
        split_neg_count = 0
        split_record_count = 0
        
        for record in records:
            category = record['category']
            attr_pair_count_pos = sum(1 for value in record['target_scores'].values() if "n/a" not in value)
            attr_pair_count_neg = sum(1 for value in record['target_scores'].values() if "n/a" in value)

            category_avg_data[split][category]['pos_count'] += attr_pair_count_pos
            category_avg_data[split][category]['neg_count'] += attr_pair_count_neg
            category_avg_data[split][category]['record_count'] += 1

            split_pos_count += attr_pair_count_pos
            split_neg_count += attr_pair_count_neg
            split_record_count += 1

        if split_record_count == 0:
            continue

        split_avg_pos = split_pos_count / split_record_count
        split_avg_neg = split_neg_count / split_record_count

        print(f"Averages for split {split} with record count {split_record_count}:")
        print(f"Average Positive Pairs: {split_avg_pos}")
        print(f"Average Negative Pairs: {split_avg_neg}")

        for category in category_avg_data[split]:
            cat_data = category_avg_data[split][category]
            cat_avg_pos = cat_data['pos_count'] / cat_data['record_count'] if cat_data['record_count'] > 0 else 0
            cat_avg_neg = cat_data['neg_count'] / cat_data['record_count'] if cat_data['record_count'] > 0 else 0
            print(f"Averages for category {category} in split {split}:")
            print(f"Average Positive Pairs: {cat_avg_pos}")
            print(f"Average Negative Pairs: {cat_avg_neg}")

    print("AVERAGE NUMBER OF PAIRS ACROSS ALL DATA")

    category_avg_data = defaultdict(lambda: {'pos_count': 0, 'neg_count': 0, 'record_count': 0})

    overall_pos_count = 0
    overall_neg_count = 0
    overall_record_count = 0

    for records in loaded_datasets.values(): 
        for record in records:
            category = record['category']
            attr_pair_count_pos = sum(1 for value in record['target_scores'].values() if "n/a" not in value)
            attr_pair_count_neg = sum(1 for value in record['target_scores'].values() if "n/a" in value)

            category_avg_data[category]['pos_count'] += attr_pair_count_pos
            category_avg_data[category]['neg_count'] += attr_pair_count_neg
            category_avg_data[category]['record_count'] += 1

            overall_pos_count += attr_pair_count_pos
            overall_neg_count += attr_pair_count_neg
            overall_record_count += 1

    if overall_record_count == 0:
        print("No records to process.")
    else:
        overall_avg_pos = overall_pos_count / overall_record_count
        overall_avg_neg = overall_neg_count / overall_record_count

        print(f"Overall record count: {overall_record_count}")
        print(f"Average Positive Pairs: {overall_avg_pos}")
        print(f"Average Negative Pairs: {overall_avg_neg}")

        for category, cat_data in category_avg_data.items():
            cat_avg_pos = cat_data['pos_count'] / cat_data['record_count'] if cat_data['record_count'] > 0 else 0
            cat_avg_neg = cat_data['neg_count'] / cat_data['record_count'] if cat_data['record_count'] > 0 else 0
            print(f"Averages for category {category}:")
            print(f"Average Positive Pairs: {cat_avg_pos}")
            print(f"Average Negative Pairs: {cat_avg_neg}")

    if dataset == "wdc":
        category_avg_data = {}
        overall_pos_count = 0
        overall_neg_count = 0
        overall_record_count = 0

        for records in loaded_datasets.values():
            for record in records:
                category = record['category']
                if category not in category_avg_data:
                    category_avg_data[category] = {'pos_counts': [], 'neg_counts': [], 'pos_count': 0, 'neg_count': 0, 'record_count': 0}
                
                attr_pair_count_pos = sum(1 for value in record['target_scores'].values() if "n/a" not in value)
                attr_pair_count_neg = sum(1 for value in record['target_scores'].values() if "n/a" in value)

                category_avg_data[category]['pos_counts'].append(attr_pair_count_pos)
                category_avg_data[category]['neg_counts'].append(attr_pair_count_neg)

                category_avg_data[category]['pos_count'] += attr_pair_count_pos
                category_avg_data[category]['neg_count'] += attr_pair_count_neg
                category_avg_data[category]['record_count'] += 1

                overall_pos_count += attr_pair_count_pos
                overall_neg_count += attr_pair_count_neg
                overall_record_count += 1

        if overall_record_count == 0:
            print("No records to process.")
        else:
            overall_avg_pos = overall_pos_count / overall_record_count
            overall_avg_neg = overall_neg_count / overall_record_count

        max_pos_count = max(max(data['pos_counts']) for data in category_avg_data.values())
        n_categories = len(category_avg_data.keys())

        fig, axs = plt.subplots(1, n_categories, figsize=(5*n_categories, 6), sharey=True)

        if n_categories == 1:
            axs = [axs]

        for ax, (category, data) in zip(axs, category_avg_data.items()):
            pos_counts = data['pos_counts']
            
            avg_pos = data['pos_count'] / data['record_count']
            avg_neg = data['neg_count'] / data['record_count']
            
            ax.hist(pos_counts, bins=10, color='skyblue', alpha=0.7, range=(0, max_pos_count+5))
            
            ax.axvline(avg_pos, color='red', linestyle='dashed', linewidth=1, label=f'Avg Pos: {avg_pos:.2f}')
            ax.axvline(avg_neg, color='black', linestyle='dashed', linewidth=1, label=f'Avg Neg: {avg_neg:.2f}')
            
            ax.set_title(f'{category}')
            ax.set_xlabel('Positive Counts')
            ax.set_ylabel('Frequency' if ax is axs[0] else '')  
            ax.set_xlim(0, max_pos_count + 5)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'../resources/WDC_histogram_positive_attribute_counts.svg')



    print("NUMBER OF UNIQUE VALUES PER ATTRIBUTE")
    attribute_values_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for split, records in loaded_datasets.items():
        for record in records:
            category = record['category']  

            for attr, value in record['target_scores'].items():

                if value != "n/a" and not isinstance(value, dict):
                    attribute_values_data[split][category][attr].add(value)
                elif isinstance(value, dict):
                    attribute_values_data[split][category][attr].add(tuple(sorted(value.items())))
        
        for category, attrs in attribute_values_data[split].items():
            total_unique_values = 0
            for values in attrs.values():
                total_unique_values += len(values)
            
            avg_unique_values = (total_unique_values / len(attrs)) if attrs else 0
            print(f"For split '{split}' in category '{category}', each attribute has on average {avg_unique_values:.2f} unique values.")

        total_unique_values_split = sum(len(values) for attrs in attribute_values_data[split].values() for values in attrs.values())
        total_attributes_split = sum(len(attrs) for attrs in attribute_values_data[split].values())
        avg_unique_values_split = (total_unique_values_split / total_attributes_split) if total_attributes_split else 0
        print(f"For split '{split}', each attribute has on average {avg_unique_values_split:.2f} unique values overall.")

    print("OVERALL")

    attribute_values_data_overall = defaultdict(lambda: defaultdict(set))

    for _, records in loaded_datasets.items():
        for record in records:
            category = record['category']
            for attr, value in record['target_scores'].items():
                if value != "n/a" and not isinstance(value, dict):
                    attribute_values_data_overall[category][attr].add(value)
                elif isinstance(value, dict):
                    attribute_values_data_overall[category][attr].add(tuple(sorted(value.items())))

    total_unique_values_overall = 0
    total_attributes_overall = 0
    unique_values_per_attribute = []  # List to hold the number of unique values per attribute
    for category, attrs in attribute_values_data_overall.items():
        for values in attrs.values():
            total_unique_values_overall += len(values)
            unique_values_per_attribute.append(len(values))  # Append the count to the list
        total_attributes_overall += len(attrs)

    avg_unique_values_overall = (total_unique_values_overall / total_attributes_overall) if total_attributes_overall else 0

    # Calculate median
    median_unique_values_overall = np.median(unique_values_per_attribute) if unique_values_per_attribute else 0
    # Calculate max
    max_unique_values_overall = max(unique_values_per_attribute) if unique_values_per_attribute else 0

    print(f"Across all data, each attribute has on average {avg_unique_values_overall:.2f} unique values.")
    print(f"The median number of unique values per attribute across all data is {median_unique_values_overall}.")
    print(f"The maximum number of unique values for a single attribute across all data is {max_unique_values_overall}.")

    for category, attrs in attribute_values_data_overall.items():
        total_unique_values_category = sum(len(values) for values in attrs.values())
        avg_unique_values_category = (total_unique_values_category / len(attrs)) if attrs else 0
        print(f"In category '{category}', each attribute has on average {avg_unique_values_category:.2f} unique values.")

    print("NUMBER OF ACCPTED VALUS PER ATTRIBUTE")
    category_averages = defaultdict(lambda: {'total_value_count': 0, 'total_attribute_count': 0})

    for records in loaded_datasets.values(): 
        for record in records:
            category = record['category']
            target_scores = record['target_scores']

            for values in target_scores.values():
                accepted_values_count = len(values)
                category_averages[category]['total_value_count'] += accepted_values_count
                category_averages[category]['total_attribute_count'] += 1

    for category, data in category_averages.items():
        if data['total_attribute_count'] > 0:  
            average_accepted_values = data['total_value_count'] / data['total_attribute_count']
            print(f"Category: {category}, Average Accepted Values per Attribute: {average_accepted_values:.2f}")
        else:
            print(f"Category: {category}, No attributes present.")

    total_annotated_values = 0
    total_annotated_values_na = 0

    for records in loaded_datasets.values():
        for record in records:
            target_scores = record['target_scores']

            for values in target_scores.values():
                # Summing up the counts for each value in the attribute
                total_annotated_values += sum(values.values())
                if "n/a" in values:
                    total_annotated_values_na += values["n/a"]  # Increment by the value of "n/a", not just by 1

    print(f"Total Number of Annotated Values: {total_annotated_values}")
    print(f"Total Number of 'n/a' Values: {total_annotated_values_na}")

    total_annotated_values_by_split = {}
    total_annotated_values_na_by_split = {}

    for split_name, records in loaded_datasets.items():
        # Initialize counts for this split
        total_annotated_values_by_split[split_name] = 0
        total_annotated_values_na_by_split[split_name] = 0

        for record in records:
            target_scores = record['target_scores']

            for values in target_scores.values():
                # Summing up the counts for each value in the attribute for the current split
                total_annotated_values_by_split[split_name] += sum(values.values())
                if "n/a" in values:
                    total_annotated_values_na_by_split[split_name] += values["n/a"]

    # After processing all data, print the results
    for split_name in loaded_datasets.keys():
        print(f"Split: {split_name}")
        print(f"Total Number of Annotated Values: {total_annotated_values_by_split[split_name]}")
        print(f"Total Number of 'n/a' Values: {total_annotated_values_na_by_split[split_name]}")


    print("NUMBER OF TOKENS PER ATTRIBUTE")

    for split, records in loaded_datasets.items():
        for record in records:
            category = record['category']
            attr_pair_count = sum(1 for value in record['target_scores'].values())
            
            if attr_pair_count == 0:
                continue
            
            if dataset == "mave_random" or dataset == "wdc":
                token_attr_pair_data['title'][category]['tokens'] += len(encoding.encode(record['input_title']))
                token_attr_pair_data['description'][category]['tokens'] += len(encoding.encode(record['input_description']))
                token_attr_pair_data['concatination'][category]['tokens'] += (len(encoding.encode(record['input_title'])) + len(encoding.encode(record['input_description'])))
            else:
                token_attr_pair_data['title'][category]['tokens'] += len(encoding.encode(record['input']))
            
            token_attr_pair_data['title'][category]['attr_pairs'] += attr_pair_count
            token_attr_pair_data['description'][category]['attr_pairs'] += attr_pair_count
            token_attr_pair_data['concatination'][category]['attr_pairs'] += attr_pair_count

    for field, categories in token_attr_pair_data.items():
        print(f"Avg. Number of tokens per attribute-value pair in {field}:")
        for category, data in categories.items():
            avg_tokens_per_pair = data['tokens'] / data['attr_pairs'] if data['attr_pairs'] else 0
            print(f"  - {category}: {avg_tokens_per_pair:.2f}")


    print(f'No. Product Offers: \t {len(loaded_datasets["train"]) + len(loaded_datasets["test"])}')

    print(f'No. Product Offers Train: \t {len(loaded_datasets["train"])}')
    print(f'No. Product Offers Test: \t {len(loaded_datasets["test"])}')

    print(f'No. Attributes: \t  {len(unique_attributes)}')
    print(f'No. Categories: \t {len(unique_categories)}')
    print(f'No. Category & Attribute Combinations: \t {len(unique_category_attribute_combinations)}')

    # Number of annotations
    print(f'Number of annotations: \t {sum(no_attributes_by_product["train"]) + sum(no_attributes_by_product["test"])}')
    n_annotations = sum(no_attributes_by_product["train"]) + sum(no_attributes_by_product["test"])
    # Number of annotations
    print(f'Number of train annotations: \t {sum(no_attributes_by_product["train"])}')
    # Number of annotations
    print(f'Number of test annotations: \t {sum(no_attributes_by_product["test"])}')
    # Number of positive annotations
    print(f'Number of positive annotations train: \t {sum(no_attributes_by_product["train"])  - no_negatives["train"]}')
    npos_train = sum(no_attributes_by_product["train"])  - no_negatives["train"]
    print(f'Number of positive annotations test: \t {sum(no_attributes_by_product["test"])  - no_negatives["test"]}')
    npos_test = sum(no_attributes_by_product["test"])  - no_negatives["test"]
    npos_total = npos_train + npos_test
    # Number of negative annotations
    print(f'Number of negative annotations train: \t {no_negatives["train"]}')
    print(f'Number of negative annotations test: \t {no_negatives["test"]}')
    nneg_total = no_negatives["train"] + no_negatives["test"]

    print('')
    print(f'Categories: \t {", ".join(list(unique_categories))}')
    #print(f'Attributes: \t {", ".join(list(unique_attributes))}')
    #print(f'Category & Attribute Combinations: \t {", ".join(list(unique_category_attribute_combinations))}')

    print('')
    for split in ['train', 'test']:
        # Products per category calculations
        products_per_category = [len(records) for records in records_by_category[split].values()]

        print(f"PRODUCTS PER CATEGORY: {products_per_category}")
        print(f'Average products per category in {split}: \t {round(sum(products_per_category) / len(products_per_category), 2)}')
        print(f'Median products per category in {split}: \t {sorted(products_per_category)[len(products_per_category) // 2]}')
        print(f'Min products per category in {split}: \t {min(products_per_category)}')
        print(f'Max products per category in {split}: \t {max(products_per_category)}')


        # No. attributes per product calculations
        print(
            f'Average no. attributes per product in {split}: \t {round(sum(no_attributes_by_product[split]) / len(no_attributes_by_product[split]), 2)}')
        print(
            f'Median no. attributes per product in {split}: \t {sorted(no_attributes_by_product[split])[len(no_attributes_by_product[split]) // 2]}')
        print(f'Min no. attributes per product in {split}: \t {min(no_attributes_by_product[split])}')
        print(f'Max no. attributes per product in {split}: \t {max(no_attributes_by_product[split])}')

        # No. attributes per category calculations
        attributes_per_category = [len(attributes) for attributes in attributes_by_category[split].values()]
        print(
            f'Average no. attributes per category in {split}: \t {round(sum(attributes_per_category) / len(attributes_per_category), 2)}')
        print(f'Median no. attributes per category in {split}: \t {sorted(attributes_per_category)[len(attributes_per_category) // 2]}')
        print(f'Min no. attributes per category in {split}: \t {min(attributes_per_category)}')
        print(f'Max no. attributes per category in {split}: \t {max(attributes_per_category)}')

        # No. annotations per category & attribute calculations
        annotations_per_category = [sum([sum(attribute.values()) for attribute in attributes.values()]) for attributes in attributes_by_category[split].values()]
        print(
            f'Average no. annotations per category in {split}: \t {round(sum(annotations_per_category) / len(annotations_per_category), 2)}')
        print(f'Median no. annotations per category in {split}: \t {sorted(annotations_per_category)[len(annotations_per_category) // 2]}')
        print(f'Min no. annotations per category in {split}: \t {min(annotations_per_category)}')
        print(f'Max no. annotations per category in {split}: \t {max(annotations_per_category)}')

        # No. annotations per category calculations
        annotations_per_category_attribute = []
        for attributes in attributes_by_category[split].values():
            annotations_per_category_attribute.extend([sum(attribute.values()) for attribute in attributes.values()])
        print(
            f'Average no. annotations per category & attribute: \t {round(sum(annotations_per_category_attribute) / len(annotations_per_category_attribute), 2)}')
        print(f'Median no. annotations per category & attribute: \t {sorted(annotations_per_category_attribute)[len(annotations_per_category_attribute) // 2]}')
        print(f'Min no. annotations per category & attribute: \t {min(annotations_per_category_attribute)}')
        print(f'Max no. annotations per category & attribute: \t {max(annotations_per_category_attribute)}')

    # Calculate unique number of attributes per category
    unique_attributes_per_category = {}
    for split in ['train', 'test']:
        unique_attributes_per_category[split] = {}
        for category in attributes_by_category[split].keys():
            if category not in unique_attributes_per_category[split]:
                unique_attributes_per_category[split][category] = set()
            for attribute in attributes_by_category[split][category].keys():
                unique_attributes_per_category[split][category].add(attribute)

    # Calculate number of attribute-value pairs in train and test
    print('')
    no_attribute_value_pairs = {'train': 0, 'test': 0}
    for split in ['train', 'test']:
        # Calculate number of products per category
        for category in records_by_category[split].keys():
            no_attribute_value_pairs[split] += len(records_by_category[split][category]) * len(unique_attributes_per_category[split])

        print(f'Number of attribute-value pairs in {split}: \t {no_attribute_value_pairs[split]}')
    print('')

    # No. attribute values per attribute calculations
    no_attribute_values_by_attribute = []
    no_normalized_attribute_values_by_attribute = []
    no_numeric_attributes = 0
    no_mostly_numeric_attributes = 0

    attribute_value_length_by_attribute = {'short': 0, 'medium': 0, 'long': 0}

    token_length_of_attribute_values = []

    # Calculate total number of attribute values
    attribute_values_by_catgory_attribute_all = {}

    for split in ['train', 'test']:
        no_attribute_values_split = 0
        for category in attribute_values_by_catgory_attribute[split].keys():
            for attribute in attribute_values_by_catgory_attribute[split][category].keys():
                if category not in attribute_values_by_catgory_attribute_all:
                    attribute_values_by_catgory_attribute_all[category] = {}
                if attribute not in attribute_values_by_catgory_attribute_all[category]:
                    attribute_values_by_catgory_attribute_all[category][attribute] = set()
                no_attribute_values_split += len(attribute_values_by_catgory_attribute[split][category][attribute])
                for attribute_value in attribute_values_by_catgory_attribute[split][category][attribute]:
                    attribute_values_by_catgory_attribute_all[category][attribute].add(attribute_value)
        print(f'Number of unique attribute values in {split}: \t {no_attribute_values_split}')

    no_attribute_values = 0
    no_values_per_attribute_category = {}
    for category in attribute_values_by_catgory_attribute_all.keys():
        for attribute in attribute_values_by_catgory_attribute_all[category].keys():
            no_attribute_values += len(attribute_values_by_catgory_attribute_all[category][attribute])
            no_values_per_attribute_category[f'{category}-{attribute}'] = len(attribute_values_by_catgory_attribute_all[category][attribute])

    # Print total number of attribute values
    print(f'Total number of attribute values: \t {no_attribute_values}')
    print(f'Average number of attribute values per category-attribute: \t {round(no_attribute_values / len(no_values_per_attribute_category), 2)}')
    print(f'Median number of attribute values per category-attribute: \t {sorted(no_values_per_attribute_category.values())[len(no_values_per_attribute_category.values()) // 2]}')

    # Calculate number of attribute values from test set that are not in train set
    no_attribute_values_test_not_in_train = 0
    for category in attribute_values_by_catgory_attribute['test'].keys():
        for attribute in attribute_values_by_catgory_attribute['test'][category].keys():
            for attribute_value in attribute_values_by_catgory_attribute['test'][category][attribute]:
                if attribute not in attribute_values_by_catgory_attribute['train'][category]:
                    no_attribute_values_test_not_in_train += 1
                elif attribute_value not in attribute_values_by_catgory_attribute['train'][category][attribute]:
                    no_attribute_values_test_not_in_train += 1

    # Print number of attribute values from test set that are not in train set
    print(f'Number of attribute values from test set that are not in train set: \t {no_attribute_values_test_not_in_train}')

    print("LIST OF UNIQUE ATTRIBUTE VALUES PER CATEGORY")
    attribute_values_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for split, records in loaded_datasets.items():
        for record in records:
            category = record['category']

            for attr, value in record['target_scores'].items():
                if value != "n/a" and not isinstance(value, dict):
                    attribute_values_data[split][category][attr].add(value)
                elif isinstance(value, dict):
                    attribute_values_data[split][category][attr].add(tuple(sorted(value.items())))


    data = {
        'Number of Products': [has_non_na_overall],
        'Number of Unique Categories': [len(unique_categories)],
        'Number of Unique Category-Attributes': [len(unique_category_attribute_combinations)],
        'Average Number of Positive Pairs per Product': [overall_avg_pos],
        'Average Number of Negative Pairs per Product': [overall_avg_neg],
        'Average Number of Unique Values per Attribute': [f"{avg_unique_values_overall:.2f}"],
        "Total Number of Annotated Values": total_annotated_values,
        "Of which are n/a": total_annotated_values_na,
        "Number of Pairs": n_annotations, # find better name
        "Of which are Positive": npos_total/n_annotations,
        "Of which are Negative": nneg_total/n_annotations
    }

    if dataset == "wdc":
        category_counts = {
        "Computers And Accessories": {"total_annotated_values": 0, "total_annotated_values_na": 0},
        "Home And Garden": {"total_annotated_values": 0, "total_annotated_values_na": 0},
        "Office Products": {"total_annotated_values": 0, "total_annotated_values_na": 0},
        "Grocery And Gourmet Food": {"total_annotated_values": 0, "total_annotated_values_na": 0},
        "Jewelry": {"total_annotated_values": 0, "total_annotated_values_na": 0}
        }

        for records in loaded_datasets.values():
            for record in records:
                target_scores = record['target_scores']
                category = record['category']

                for values in target_scores.values():
                    total_annotated_values = sum(values.values())
                    total_annotated_values_na = values.get("n/a", 0)

                    category_counts[category]["total_annotated_values"] += total_annotated_values
                    category_counts[category]["total_annotated_values_na"] += total_annotated_values_na

        for category, counts in category_counts.items():
            print(f"Category: {category}")
            print(f"Total Number of Annotated Values: {counts['total_annotated_values']}")
            print(f"Of which are n/a: {counts['total_annotated_values_na'] / counts['total_annotated_values']}")
            print()

        df = pd.DataFrame(data)
        transposed_df = df.T
        transposed_df.reset_index(inplace=True)
        transposed_df.columns = ['Stat', 'Value']
        print(transposed_df)

        unique_urls = {}
        for split, records in loaded_datasets.items():
            for record in records:
                category = record['category']
                url = record["url"]
                if category not in unique_urls:
                    unique_urls[category] = set()

                unique_urls[category].add(url)

        print("Number of Unique URLs")
        for category, urls in unique_urls.items():
            print(f"{category}: {len(urls)}")

        products_per_category = {}

        for split, records in loaded_datasets.items():
            for record in records:
                category = record['category']
                if category in products_per_category:
                    products_per_category[category] += 1
                else:
                    products_per_category[category] = 1

        for category, count in products_per_category.items():
            print(f"Category: {category}, Number of Products: {count}")

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
