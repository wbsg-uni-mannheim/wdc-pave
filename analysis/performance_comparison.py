import gzip
import json
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def error_analysis(path_task_dict_1, path_task_dict_2, with_normalization):
    with open(path_task_dict_1, 'r') as f:
        task_dict1 = json.load(f)

    with open(path_task_dict_2, 'r') as f:
        task_dict2 = json.load(f)

    def extract_f1_scores(task_dict):
        """Extract F1 scores from the task dictionary, ignoring non-attribute entries."""
        f1_scores = {}
        for key, value in task_dict['results'].items():
            parts = key.split('__')
            if len(parts) == 2:  
                attribute, category = parts
                if category not in f1_scores:
                    f1_scores[category] = {}
                f1_scores[category][attribute] = value['f1']
        return f1_scores

    def compare_f1_scores(f1_before, f1_after):
        """Compare F1 scores and calculate the differences."""
        comparison_dict = {}
        for category in f1_before:
            comparison_dict[category] = {}
            for attribute in f1_before[category]:
                if attribute in f1_after[category]:
                    before_score = f1_before[category][attribute]
                    after_score = f1_after[category][attribute]
                    comparison_dict[category][attribute] = after_score - before_score
        return comparison_dict

    def plot_f1_changes(comparison_dict):
        """Plot the changes in F1 scores."""
        for category, attributes in comparison_dict.items():
            attributes_sorted = sorted(attributes.items(), key=lambda x: x[1]) 
            labels = [attr for attr, _ in attributes_sorted]
            changes = [change for _, change in attributes_sorted]

            plt.figure(figsize=(10, len(labels) * 0.5))
            plt.barh(labels, changes, color=np.where(np.array(changes) > 0, 'g', 'r'))
            plt.xlabel('F1 Score Change')
            plt.title(f'F1 Score Changes in {category}')
            plt.grid(axis='x')

            plt.savefig(f'../figures/change_in_f1_scores_{category}.svg')
            plt.close()

    f1_scores_before = extract_f1_scores(task_dict1)
    f1_scores_after = extract_f1_scores(task_dict2)

    comparison_dict = compare_f1_scores(f1_scores_before, f1_scores_after)

    for category, attributes in comparison_dict.items():
        for attribute, difference in attributes.items():
            print(f"{category}, {attribute} Difference: {difference}")
    plot_f1_changes(comparison_dict)

    if with_normalization:
        directory_path = "../data/descriptions/wdc/descriptions.csv"
        descriptions_csv = pd.read_csv(directory_path, sep=";")

        descriptions_csv["Normalization_params"] = descriptions_csv["Normalization_params"].str.strip("[]").str.replace("'", "")
        descriptions_csv["Normalization_params_general"] = descriptions_csv["Normalization_params_general"].str.strip("[]").str.replace("'", "")

        attribute_normalization_map = {}
        for _, row in descriptions_csv.iterrows():
            category = row['Category'].strip()  
            attribute = row['Attribute'].strip()
            normalization = row['Normalization_params_general']
            
            category_attribute_key = f"{category}__{attribute}"
            attribute_normalization_map[category_attribute_key] = normalization

        enhanced_comparison_dict = {}
        for category, attributes in comparison_dict.items():
            for attribute, difference in attributes.items():

                category_attribute_key = f"{category}__{attribute}"
                
                normalization = attribute_normalization_map.get(category_attribute_key, "No Normalization")
                
                if normalization not in enhanced_comparison_dict:
                    enhanced_comparison_dict[normalization] = {}
                if category not in enhanced_comparison_dict[normalization]:
                    enhanced_comparison_dict[normalization][category] = {}
                enhanced_comparison_dict[normalization][category][attribute] = difference

        for normalization, categories in enhanced_comparison_dict.items():
            print(f"Normalization: {normalization}")
            for category, attributes in categories.items():
                for attribute, difference in attributes.items():
                    print(f"\tCategory: {category}, Attribute: {attribute}, Difference: {difference}")

        def plot_f1_differences_for_operation(normalization, categories_attributes):
            num_categories = len(categories_attributes)
            fig, axes = plt.subplots(num_categories, 1, figsize=(10, num_categories * 5), constrained_layout=True)
            
            if num_categories == 1:
                axes = [axes]
            
            for ax, (category, attributes) in zip(axes, categories_attributes.items()):
                sorted_attributes = sorted(attributes.items(), key=lambda x: x[1])
                attribute_names = [attr for attr, _ in sorted_attributes]
                differences = [diff for _, diff in sorted_attributes]
                
                ax.barh(attribute_names, differences, color=np.where(np.array(differences) > 0, 'g', 'r'))
                
                ax.set_title(f'Category: {category}')
                ax.set_xlabel('F1 Score Difference')
                ax.set_ylabel('Attribute')
                ax.axvline(0, color='grey', linewidth=0.8)  
                
            fig.suptitle(f'Normalization Operation: {normalization}', fontsize=16)

            plt.savefig(f'../figures/change_in_f1_scores_{normalization}_normalization.svg')
            plt.show()

    for normalization, categories_attributes in enhanced_comparison_dict.items():
        plot_f1_differences_for_operation(normalization, categories_attributes)

if "__main__" == __name__:
    path_task_dict_1 = "../prompts/runs/"
    path_task_dict_2 = "../prompts/runs/"
    error_analysis(path_task_dict_1, path_task_dict_2, with_normalization=True)
