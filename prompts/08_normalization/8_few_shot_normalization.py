import json
import random
from datetime import datetime
from json import JSONDecodeError
import ast

import click
import torch
from dotenv import load_dotenv
from langchain import LLMChain, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import SystemMessage
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import create_pydanctic_models_from_known_attributes, parse_llm_response_to_json, save_populated_task_to_json, get_normalization_guidelines_from_csv
from pieutils.evaluation import evaluate_predictions, visualize_performance, calculate_cost_per_1k
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_normalized_test_set
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector
from pieutils.normalization import normalize_data


@click.command()
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='MaxMarginalRelevance', help='Example selector for few-shot learning')
@click.option('--dataset', default='wdc', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for example value selection')
@click.option('--title', default=False, help = 'Include Title')
@click.option('--description', default=False, help = 'Include description')
@click.option('--normalization_params', default="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Binary Classification', 'Name Generalisation','Unit Expansion', 'To Uppercase', 'Delete Marks']", help = 'Which normalizations should be included')
@click.option('--normalized_only', default=True, help = 'Extract only attributes that are viable for normalization')
@click.option('--force_from_different_website', default=False, help = 'For WDC Data, ensures that shots come from different url')
def main(dataset, model, verbose, shots, example_selector, train_percentage, title, description, normalization_params, normalized_only, force_from_different_website):
    
    # Load task template
    with open('../prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    normalization_params = ast.literal_eval(normalization_params)

    normalized_dataset_name = f"{dataset}_normalized"

    task_dict['task_name'] = f"few_shot_normalization_only_{shots}_shots_{train_percentage}_train_percentage"
    task_dict['task_prefix'] = "Split the product {part} by whitespace. " \
                                    "Extract the valid attribute values from the product {part} and normalize the " \
                                    "attribute values according to the guidelines below. Respond in JSON format. Unknown attribute values should be marked as n/a. Do not explain your answer. \n" \
                                    "Guidelines: \n" \
                                    "{guidelines}"

    task_dict['data_source'] = dataset
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Normalize the data
    train_split = f"train_{train_percentage}"
    normalized_attributes = normalize_data(split=train_split, normalization_params=normalization_params)
    normalized_attributes = normalize_data(split="test", normalization_params=normalization_params)

    print(normalized_attributes)


    # Load examples into task dict
    params = "_".join(normalization_params)
    file_name_test = f"normalized_test_{params}"
    task_dict = update_task_dict_from_normalized_test_set(task_dict,file_name_test,title, description, normalized_only, normalized_attributes)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0, request_timeout=120)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    save_populated_task_to_json(task_dict['task_name'], task_dict, title, description)

    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")

    # Add few shot sampling
    if example_selector == 'SemanticSimilarity':
        category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(normalized_dataset_name, 
                                                                                     list(task_dict['known_attributes'].keys()),
                                                                                     title, description,
                                                                                     load_from_local=False, k=shots, train_percentage=train_percentage, normalization_params=normalization_params, normalized_only=normalized_only, normalized_attributes=normalized_attributes, force_from_different_website = force_from_different_website)

    else:
        raise NotImplementedError("Example selector {} not implemented".format(example_selector))

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input", "category", "url"],
        example_selector=category_example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    prompt = ChatPromptTemplate(messages=[system_setting, 
                                          human_task_prompt, 
                                          few_shot_prompt,
                                          human_task_prompt,
                                          human_message_prompt])

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    def select_and_run_llm(category, input_text, part='title', url=None):

        guidelines = get_normalization_guidelines_from_csv(normalization_params, category, normalized_only)

        pred = None
        try:
            response = chain.run(input=input_text, attributes=', '.join(task_dict['known_attributes'][category]),
                             category=category, part=part, guidelines=guidelines, url=url)
            if verbose:
                print(response)
            try:
                # Convert json string into pydantic model
                json_response = json.loads(response)
                if verbose:
                    print(json_response)
                pred = pydantic_models[category](**json.loads(response))
            except ValidationError as valError:
                print(valError)
            except JSONDecodeError as e:
                converted_response = parse_llm_response_to_json(response)
                if verbose:
                    print('Converted response: ')
                    print(converted_response)
                try:
                    pred = pydantic_models[category](**converted_response)
                except ValidationError as valError:
                    print(valError)
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)
                print('Response: ')
                print(response)
        except Exception as e:
            print(e)


        return pred

    with get_openai_callback() as cb:
        if dataset == 'wdc':
            if title and description:
                # append title and description
                preds = [select_and_run_llm(example['category'], 
                                            f"title: {example['input_title']} \n description: {example['input_description']}", 
                                            part='title and description', url=example["url"])
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]
            elif title and not description:
                preds = [select_and_run_llm(example['category'], 
                                            f"title: {example['input_title']}", 
                                            part='title')
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]
            elif description and not title:
                preds = [select_and_run_llm(example['category'], 
                                            f"description: {example['input_description']}", 
                                            part='description')
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]

        else:
            preds = [select_and_run_llm(example['category'], example['input']) for example in
                    tqdm(task_dict['examples'], disable=not verbose)]

        task_dict['total_tokens'] = cb.total_tokens
        task_dict['prompt_tokens'] = cb.prompt_tokens
        task_dict['completion_tokens'] = cb.completion_tokens
        task_dict['total_costs'] = cb.total_cost
        total_costs=cb.total_cost
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost: {cb.total_cost}")

    # Calculate recall, precision and f1
    task_dict['results'] = evaluate_predictions(preds, task_dict)
    task_dict['costs_per_1000'] = calculate_cost_per_1k(total_costs, task_dict)

    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task_to_json(task_dict['task_name'], task_dict, title, description)
    # Visualize results
    visualize_performance(task_dict)

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
