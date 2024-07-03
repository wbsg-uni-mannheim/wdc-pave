from collections import defaultdict
import pandas as pd
import json
from tqdm import tqdm
import os
import click
from dotenv import load_dotenv

@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')

def main(dataset):
    if dataset == "wdc":
        directory_path = "../data/raw/wdc/"
        product_dict = []
        for filename in tqdm(sorted(os.listdir(directory_path))):
            with open(os.path.join(directory_path, filename)) as file:
                total_lines = sum(1 for line in file)
                file.seek(0) 
                for line in tqdm(file, total=total_lines, desc=f'Processing {filename}', unit='line'):
                    product_dict.append(json.loads(line))

        df = pd.DataFrame(product_dict)
        directory_path = "../data/raw/wdc/"
        product_dict = []
        for filename in tqdm(sorted(os.listdir(directory_path))):
            with open(os.path.join(directory_path, filename)) as file:
                total_lines = sum(1 for line in file)
                file.seek(0) 
                for line in tqdm(file, total=total_lines, desc=f'Processing {filename}', unit='line'):
                    product_dict.append(json.loads(line))

        df = pd.DataFrame(product_dict)

    elif dataset == "wdc_normalized":
        directory_path_train = f"../data/processed_datasets/deprecated/wdc/normalized/normalized_train_1.0_Name Expansion_Numeric Standardization_To Uppercase_Selective Identifier Parsing_Product Type Generalisation_Unit Conversion_Color Generalization_Binary Classification_Name Generalisation_Unit Expansion_To Uppercase_Delete Marks.jsonl"
        directory_path_test = f"../data/processed_datasets/deprecated/wdc/normalized/normalized_test_Name Expansion_Numeric Standardization_To Uppercase_Selective Identifier Parsing_Product Type Generalisation_Unit Conversion_Color Generalization_Binary Classification_Name Generalisation_Unit Expansion_To Uppercase_Delete Marks.jsonl"
        directories = [directory_path_train, directory_path_test]

        product_dict = []
        for filename in directories:
            with open(filename, 'r') as file:
                total_lines = sum(1 for line in file)
                file.seek(0) 
                for line in tqdm(file, total=total_lines, desc=f'Processing {filename}', unit='line'):
                    product_dict.append(json.loads(line))
        
        df = pd.DataFrame(product_dict)

    
    else:
        directory_path_train = f"../data/processed_datasets/{dataset}/train.jsonl"
        directory_path_test = f"../data/processed_datasets/{dataset}/train.jsonl"
        directories = [directory_path_train, directory_path_test]

        product_dict = []
        for filename in directories:
            with open(filename, 'r') as file:
                total_lines = sum(1 for line in file)
                file.seek(0) 
                for line in tqdm(file, total=total_lines, desc=f'Processing {filename}', unit='line'):
                    product_dict.append(json.loads(line))
        
        df = pd.DataFrame(product_dict)

    def flatten(items):
        """Flatten a list of items, including nested lists."""
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    def format_percentage(value):
        return f"{value:.2f}%"

    def summarize_attributes(df):
        # Dictionaries to store the aggregated data
        aggregated_data_scores = defaultdict(lambda: defaultdict(list))

        # Process each row in the DataFrame
        for _, row in df.iterrows():
            category = row['category']
            
            # Handling target_scores
            target_scores = row['target_scores']
            if isinstance(target_scores, str):
                target_scores = json.loads(target_scores)
            for attribute, value_dict in target_scores.items():
                for value, details in value_dict.items():
                    aggregated_data_scores[category][attribute].append(value)

        # List to store the final summary data
        summary_data = []

        # Generating summary statistics for each category-attribute pair
        for category, attributes in aggregated_data_scores.items():
            for attribute, values in attributes.items():
                unique_values_scores = set(flatten(values))
                na_count_scores = values.count('n/a')
                total_count_scores = len(list(flatten(values)))

                summary_data.append({
                    'Category': category,
                    'Attribute': attribute,
                    'Unique Values Count (target_scores)': len(unique_values_scores),
                    'NA Values Percentage (target_scores)': format_percentage((na_count_scores / total_count_scores) * 100) if total_count_scores else '0.00%',
                    'Unique Values List (target_scores)': ', '.join(map(str, unique_values_scores)),
                })

        # Convert the summary data to a DataFrame and return
        return pd.DataFrame(summary_data)


    summary_df = summarize_attributes(df)
    summary_df.to_csv(f"../data/descriptions/{dataset}/summary_statistics_{dataset}.csv", encoding="utf-8-sig")

    if dataset == "wdc_normalized":
        records = df.to_dict(orient='records')
        with open(f"../data/raw/wdc_normalized/normalized_target_scores.jsonl", "w", encoding="utf-8") as file:
            for record in records:
                if 'target_scores' in record and isinstance(record['target_scores'], str):
                    try:
                        record['target_scores'] = json.loads(record['target_scores'])
                    except json.JSONDecodeError:
                        print("Error")  
                json_record = json.dumps(record, ensure_ascii=False)
                file.write(json_record + '\n')

if __name__ == '__main__':
    load_dotenv()
    main()