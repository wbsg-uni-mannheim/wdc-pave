import json
import os
import random
from typing import List, Dict

import faiss
import torch
from dotenv import load_dotenv
from langchain import FewShotPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

#from config import FAISS_INDEXES, PROCESSED_DATASETS
from pieutils.config import FAISS_INDEXES, PROCESSED_DATASETS

def load_train_for_vector_store(dataset_name, categories, title, description, train_percentage=1.0, normalization_params=None, normalized_only=False,normalized_attributes=None, force_from_different_website=False):
    """Load dataset train records for vector store.
        If categories is specified the subset is considered otherwise all categories are loaded."""

    if dataset_name == "wdc_normalized":
        params = "_".join(normalization_params)
        name = f"normalized_train_{train_percentage}_{params}"
        directory_path = f'{PROCESSED_DATASETS}/wdc/normalized'
        file_path = os.path.join(directory_path, f'{name}.jsonl')

        if normalized_only:
            train_records = {}
            example_id = 0
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    url = record.get('url') if force_from_different_website else None
                    if dataset_name in ['wdc_normalized']:
                        for part in ['input_title', 'input_description', 'target_scores']:
                            if part in record:
                                if categories is None or record['category'] in categories:
                                    if part == "target_scores":
                                        if description and title:
                                            input_data = f"{record.get('input_title', '')} {record.get('input_description', '')}".strip()
                                        elif title:
                                            input_data = record.get('input_title', '') 
                                        elif description:
                                            input_data = record.get('input_description', '') 

                                        category = record['category']
                                        normalized_attrs = normalized_attributes.get(category, [])

                                        train_records[str(example_id)] = {
                                            'input': input_data,
                                            'category': category,
                                            'part': part,
                                            'url': url,
                                            'extractions': {}
                                        }

                                        for key, value_dict in record['target_scores'].items():
                                            if key in normalized_attrs:
                                                if 'n/a' in value_dict:
                                                    train_records[str(example_id)]['extractions'][key] = "n/a"
                                                else:
                                                    contained_values = []
                                                    for value, details in value_dict.items():
                                                        pid_list = details.get('pid', [])
                                                        if title and description and any(pid in pid_list for pid in [0, 1]):
                                                            contained_values.append(value)
                                                        elif title and not description and 0 in pid_list:
                                                            contained_values.append(value)
                                                        elif description and not title and 1 in pid_list:
                                                            contained_values.append(value)

                                                    if len(contained_values) > 0:
                                                        train_records[str(example_id)]['extractions'][key] = max(contained_values, key=len)
                                                    else:
                                                        train_records[str(example_id)]['extractions'][key] = "n/a"
                                        example_id += 1

                filtered_train_records = {}
                for record_id, record in train_records.items():
                    if any(value != "n/a" for value in record['extractions'].values()):
                        filtered_train_records[record_id] = record

                train_records = filtered_train_records
                

        else:
            train_records = {}
            example_id = 0
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    # Load record from title, description, features and rest of the attributes. - Mainly relevant for MAVE
                    record = json.loads(line)
                    url = record.get('url') if force_from_different_website else None
                    if dataset_name in ['wdc_normalized']:
                        for part in ['input_title', 'input_description', 'target_scores']:
                            if part in record:
                                if categories is None or record['category'] in categories:
                                    if part == "target_scores":
                                        if description and title:
                                            input_data = f"{record.get('input_title', '')} {record.get('input_description', '')}".strip()  # concatenating title and description
                                        elif title:
                                            input_data = record.get('input_title', '') 
                                        elif description:
                                            input_data = record.get('input_description', '') 
                                        train_records[str(example_id)] = {'input': input_data,
                                                                        'category': record['category'],
                                                                        'part': part,
                                                                        'url': url,
                                                                        'extractions': {}}
                                        for key, value_dict in record['target_scores'].items():
                                            if 'n/a' in value_dict:
                                                        train_records[str(example_id)]['extractions'][key] = "n/a"
                                            else:
                                                contained_values = []
                                                for value, details in value_dict.items():
                                                    pid_list = details.get('pid', [])
                                                    if title and description and any(pid in pid_list for pid in [0, 1]):
                                                        train_records[str(example_id)]['extractions'][key] = value
                                                        contained_values.append(value)
                                                    elif title and not description and 0 in pid_list:
                                                        train_records[str(example_id)]['extractions'][key] = value
                                                        contained_values.append(value)
                                                    elif description and not title and 1 in pid_list:
                                                        train_records[str(example_id)]['extractions'][key] = value
                                                        contained_values.append(value)

                                                if len(contained_values) > 0:
                                                    train_records[str(example_id)]['extractions'][key] = max(contained_values, key=len)
                                                else:
                                                    train_records[str(example_id)]['extractions'][key] = "n/a"
                                        example_id += 1
                        filtered_train_records = {}
                        for record_id, record in train_records.items():
                            if any(value != "n/a" for value in record['extractions'].values()):
                                filtered_train_records[record_id] = record

                        train_records = filtered_train_records

    else:
        directory_path = f'{PROCESSED_DATASETS}/{dataset_name}/'
        if train_percentage < 1.0:
            file_path = os.path.join(directory_path, f'train_{train_percentage}.jsonl')
        else:
            file_path = os.path.join(directory_path, 'train.jsonl')

        # Structure of train records: {id: {'category': category, 'input': input, 'extractions': {key: value}}}
        train_records = {}
        example_id = 0
        with open(file_path, 'r') as f:
            for line in f.readlines():
                # Load record from title, description, features and rest of the attributes. - Mainly relevant for MAVE
                record = json.loads(line)
                url = record.get('url') if force_from_different_website else None
                #print(record)
                if dataset_name in ["mave_random", 'wdc']:
                    for part in ['input_title', 'input_description', 'target_scores']:
                        if part in record:
                            if categories is None or record['category'] in categories:
                                if part == "target_scores":
                                    if description and title:
                                        input_data = f"{record.get('input_title', '')} {record.get('input_description', '')}".strip()  # concatenating title and description
                                    elif title:
                                        input_data = record.get('input_title', '') 
                                    elif description:
                                        input_data = record.get('input_description', '') 
                                    train_records[str(example_id)] = {'input': input_data,
                                                                    'category': record['category'],
                                                                    'part': part,
                                                                    'url': url,
                                                                    'extractions': {}}
                                    for key, value_dict in record['target_scores'].items():
                                        if 'n/a' in value_dict:
                                                    train_records[str(example_id)]['extractions'][key] = "n/a"
                                        else:
                                            contained_values = []
                                            for value, details in value_dict.items():
                                                pid_list = details.get('pid', [])
                                                if title and description and any(pid in pid_list for pid in [0, 1]):
                                                    train_records[str(example_id)]['extractions'][key] = value
                                                    contained_values.append(value)
                                                elif title and not description and 0 in pid_list:
                                                    train_records[str(example_id)]['extractions'][key] = value
                                                    contained_values.append(value)
                                                elif description and not title and 1 in pid_list:
                                                    train_records[str(example_id)]['extractions'][key] = value
                                                    contained_values.append(value)

                                            if len(contained_values) > 0:
                                                train_records[str(example_id)]['extractions'][key] = max(contained_values, key=len)
                                            else:
                                                train_records[str(example_id)]['extractions'][key] = "n/a"
                                    example_id += 1
                    filtered_train_records = {}
                    for record_id, record in train_records.items():
                        if any(value != "n/a" for value in record['extractions'].values()):
                            filtered_train_records[record_id] = record

                    train_records = filtered_train_records
                    #print(train_records)
                                                
                else:
                    for part in ['input', 'description', 'features', 'rest', 'paragraphs']:
                        if part in record:
                            if categories is None or record['category'] in categories:
                                if part == 'paragraphs':
                                    for paragraph in record[part]:
                                        train_records[str(example_id)] = {'input': paragraph,
                                                                        'category': record['category'],
                                                                        'part': part,
                                                                        'extractions': {}}
                                        for key, value_dict in record['target_scores'].items():
                                            if 'n/a' in value_dict:
                                                train_records[str(example_id)]['extractions'][key] = "n/a"
                                            else:
                                                contained_values = [k for k, v in value_dict.items() if k in paragraph]
                                                if len(contained_values) > 0:
                                                    train_records[str(example_id)]['extractions'][key] = max(contained_values,
                                                                                                            key=len)
                                                else:
                                                    train_records[str(example_id)]['extractions'][key] = "n/a"
                                        example_id += 1
                                else:
                                    train_records[str(example_id)] = {'input': record[part], 'part': part,
                                                                    'category': record['category'],
                                                                    'extractions': {}}
                                    for key, value_dict in record['target_scores'].items():
                                        if 'n/a' in value_dict:
                                            train_records[str(example_id)]['extractions'][key] = "n/a"
                                        else:
                                            contained_values = [k for k, v in value_dict.items() if k in record[part]]
                                            if len(contained_values) > 0:
                                                train_records[str(example_id)]['extractions'][key] = max(contained_values,
                                                                                                        key=len)
                                            else:
                                                train_records[str(example_id)]['extractions'][key] = "n/a"
                                    example_id += 1
    #print(train_records)
    return train_records


def initialize_vector_store(dataset_name, title, description, load_from_local=False, categories=None, train_percentage=1.0, normalization_params=None,normalized_only=False, normalized_attributes=None, force_from_different_website=False):
    # Load the training data.
    if dataset_name in ['mave', 'mave_v2', 'ae-110k', 'oa-mine', 'mave_random', 'wdc']:
        train_records = load_train_for_vector_store(dataset_name, categories, title, description, train_percentage=train_percentage, force_from_different_website=force_from_different_website)
    elif dataset_name == "wdc_normalized":
        train_records = load_train_for_vector_store(dataset_name, categories, title, description, train_percentage=train_percentage, normalization_params=normalization_params,normalized_only=normalized_only, normalized_attributes=normalized_attributes, force_from_different_website=force_from_different_website)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported!')

    categories = list(set([record['category'] for record in train_records.values()]))
    category_to_vector_store = {}
    # Load the product titles along with metadata into the vector store.

    if load_from_local:
        for category in categories:
            folder_path = f'{FAISS_INDEXES}/{dataset_name}'
            index_name = f'{dataset_name}_faiss_index_{category}'
            category_to_vector_store[category] = FAISS.load_local(folder_path=folder_path,
                                                                  embeddings=OpenAIEmbeddings(), index_name=index_name)
    else:
        for category in categories:
            if force_from_different_website:
                inputs = [record['input'] for record in train_records.values() if record['category'] == category]
                metadatas = [{'input': record['input'], 'output': json.dumps(record['extractions']), 'url': record['url']}
                            for record in train_records.values() if record['category'] == category]
                ids = [key for key in train_records.keys() if train_records[key]['category'] == category]
                category_to_vector_store[category] = FAISS.from_texts(inputs, OpenAIEmbeddings(), metadatas=metadatas, ids=ids)
            else:
                inputs = [record['input'] for record in train_records.values() if record['category'] == category]
                metadatas = [{'input': record['input'], 'output': json.dumps(record['extractions'])}
                            for record in train_records.values() if record['category'] == category]
                ids = [key for key in train_records.keys() if train_records[key]['category'] == category]
                category_to_vector_store[category] = FAISS.from_texts(inputs, OpenAIEmbeddings(), metadatas=metadatas,
                                                                    ids=ids)
            # Check if directory for vector store exists.
            folder_path = f'{FAISS_INDEXES}/{dataset_name}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Save the vector store locally.
            index_name = f'{dataset_name}_faiss_index_{category}'
            category_to_vector_store[category].save_local(folder_path=folder_path, index_name=index_name)
    return category_to_vector_store


def apply_pydantic_model(extractions, pydantic_model):
    """Apply the pydantic model and return the json string."""
    pydantic_dict = json.loads(pydantic_model(**extractions).json())
    for key in pydantic_dict:
        if pydantic_dict[key] is None:
            pydantic_dict[key] = 'n/a'

    pydantic_json = json.dumps(pydantic_dict)
    return pydantic_json


def cluster_examples_using_kmeans(examples, max_cluster_size=5, sliding_wino_size=0):
    """Cluster examples using k-means clustering. Assign each example to the cluster with the nearest centroid."""

    titles = [example['input'] for example in examples]

    emb = OpenAIEmbeddings(chunk_size=1)
    embeddings = emb.embed_documents(titles)
    tensor_of_text_embeddings = torch.tensor(embeddings)

    # Cluster the examples.
    d = tensor_of_text_embeddings.shape[1]
    nclusters = len(examples) // max_cluster_size if len(examples) % max_cluster_size == 0 else len(
        examples) // max_cluster_size + 1
    niter = 20
    kmeans = faiss.Kmeans(d, nclusters, niter=niter, verbose=True, min_points_per_centroid=max_cluster_size, seed=42)
    kmeans.train(tensor_of_text_embeddings)

    # Get the cluster assignments.
    D, I = kmeans.index.search(tensor_of_text_embeddings, 1)

    # clustering = AgglomerativeClustering(n_clusters=nclusters, metric='cosine', linkage='average').fit(embeddings)

    # Assign each example to the cluster with the nearest centroid.
    examples = [{'input': example['input'], 'category': example['category'],
                 'cluster_id': str(cluster[0]) if 'cluster_id' not in example else example['cluster_id'] + str(
                     cluster[0])} for example, cluster in zip(examples, I)]

    # Add a sliding window over large clusters.
    if sliding_wino_size > 0:
        for cluster_id in [example['cluster_id'] for example in examples]:
            clustered_examples = [example for example in examples if example['cluster_id'] == cluster_id]
            if len(clustered_examples) > max_cluster_size:
                # Apply sliding window over large clusters.
                # Sort examples by input.
                clustered_examples = sorted(clustered_examples, key=lambda x: x['input'])
                # Apply sliding window.
                clustered_example_windows = [clustered_examples[i:i + max_cluster_size] for i in
                                             range(0, len(clustered_examples), max_cluster_size)]
                for i, clustered_example_window in enumerate(clustered_example_windows):
                    for j, example in enumerate(clustered_example_window):
                        example['cluster_id'] = example['cluster_id'] + str(i)
                clustered_examples = [example for clustered_example_window in clustered_example_windows for example in
                                      clustered_example_window]
                for original_example, clustered_example in zip(
                        [example for example in examples if example['cluster_id'] == cluster_id], clustered_examples):
                    original_example['cluster_id'] = clustered_example['cluster_id']

    return examples


def cluster_examples_using_sliding_window(examples, max_cluster_size=5):
    """Cluster examples using a sliding window."""
    clustered_examples = sorted(examples, key=lambda x: x['input'])
    # Apply sliding window.
    clustered_example_windows = [clustered_examples[i:i + max_cluster_size] for i in
                                 range(0, len(clustered_examples), max_cluster_size)]
    for i, clustered_example_window in enumerate(clustered_example_windows):
        for j, example in enumerate(clustered_example_window):
            example['cluster_id'] = str(i)
    clustered_examples = [example for clustered_example_window in clustered_example_windows for example in
                          clustered_example_window]

    return clustered_examples


def test_clustering():
    directory_path = 'data/oa_mine/annotations/'
    examples_for_clustering = {}
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            category = ' '.join([token.capitalize() for token in filename.replace('.jsonl', '').split('_')])
            if category not in examples_for_clustering:
                examples_for_clustering[category] = []
            print('Load records for category: {}'.format(category))
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                for line in f.readlines()[:50]:
                    record = json.loads(line)
                    example = {'input': record['title'], 'category': category}
                    examples_for_clustering[category].append(example)

    # Cluster examples
    # examples = cluster_examples_using_sliding_window(examples_for_clustering['Breakfast Cereal'], 5)
    examples = cluster_examples_using_kmeans(examples_for_clustering['Breakfast Cereal'], 5)

    assert len(examples) == len(examples_for_clustering['Breakfast Cereal'])
    for cluster_id in list(set([example['cluster_id'] for example in examples])):
        clustered_examples = [example['input'] for example in examples if example['cluster_id'] == cluster_id]
        print('Cluster {}: \n {}'.format(cluster_id, '\n '.join(clustered_examples)))


def convert_example_to_pydantic_model_example(example, pydantic_model):
    """Convert example to pydantic model."""
    pydantic_example = pydantic_model(**json.loads(example['output']))
    pydantic_example_json = pydantic_example.json(indent=4).replace('null', '"n/a"')
    return {'input': example['input'], 'output': pydantic_example_json}

class CategoryAwareSemanticSimilarityExampleSelector(BaseExampleSelector):

    def __init__(self, dataset_name, categories, title, description, category_2_pydantic_models=None, load_from_local=False, k=5,
                 tabular=False, train_percentage=1.0, normalization_params=None, normalized_only=False, normalized_attributes=None, force_from_different_website=False, attributes=None) -> None:
        """Initialize the example selector from the datasets"""
        category_to_vector_store = initialize_vector_store(dataset_name, title, description, load_from_local=load_from_local, categories=categories, train_percentage=train_percentage, normalization_params=normalization_params, normalized_only=normalized_only, normalized_attributes=normalized_attributes, force_from_different_website=force_from_different_website)

        self.force_from_different_website =force_from_different_website
        self.attributes = attributes
        if force_from_different_website:
            choose_k = k * 50
            self.k = k
            self.k_to_keep = k
        else:
            self.k = k
            choose_k = k
        # Initialize Different example selectors for each category.
        self.example_selector_by_category = {
            category: SemanticSimilarityExampleSelector(vectorstore=category_to_vector_store[category], k=choose_k) for
            category in category_to_vector_store}
        self.category_2_pydantic_models = category_2_pydantic_models
        if self.category_2_pydantic_models is not None:
            self.tabular = tabular
        elif tabular:
            print('Tabular is set to True but no pydantic models are provided. Tabular will be set to False.')
            self.tabular = False

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.example_selector_by_category[example['category']].add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        category = input_variables['category']
        input_url = input_variables["url"]
        selected_examples = []

        if self.force_from_different_website:
            # Get all semantically similar examples
            all_similar_examples = self.example_selector_by_category[category].select_examples(
                {'input': input_variables['input']})

            # Filter out examples from the same URL as the input example
            different_url_examples = [example for example in all_similar_examples if example['url'] != input_url]

            for example in different_url_examples:
                if len(selected_examples) < self.k_to_keep:
                    selected_examples.append(example)
                else:
                    break

        else:
            # Select top k semantically similar examples without URL filtering
            selected_examples = self.example_selector_by_category[category].select_examples(
                {'input': input_variables['input']})[:self.k]

        # If running similar concepts or one by one, we need to eliminate all attributes not in task_dict
        if self.attributes:
            filtered_examples = []
            attributes_set = set(attribute for sublist in self.attributes for attribute in sublist)  
            for example in selected_examples:
                output_dict = json.loads(example['output'])  
                filtered_output = {key: output_dict[key] for key in attributes_set if key in output_dict}
                example['output'] = json.dumps(filtered_output) 
                filtered_examples.append(example)
            selected_examples = filtered_examples
    
        if self.category_2_pydantic_models is not None:

            if self.tabular:
                # Convert examples to tabular format.
                inputs = '\n'.join([example['input'] for example in selected_examples])
                selected_outputs = {'records': [json.loads(example['output']) for example in selected_examples]}
                pydantic_examples = self.category_2_pydantic_models[category](**selected_outputs)
                pydantic_example_json = pydantic_examples.json(indent=4).replace('null', '"n/a"')
                selected_examples = [{'inputs': inputs, 'outputs': pydantic_example_json}]
            else:
                # Convert examples to pydantic models.
                pydantic_model = self.category_2_pydantic_models[input_variables['category']]
                selected_examples = [convert_example_to_pydantic_model_example(example, pydantic_model) for example in
                                    selected_examples]

        return selected_examples


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)