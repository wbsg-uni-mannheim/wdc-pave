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
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task_to_json ,parse_llm_response_to_json, get_normalization_guidelines_from_csv
from pieutils.evaluation import evaluate_predictions, calculate_cost_per_1k, visualize_performance
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_normalized_test_set
from pieutils.normalization import normalize_data


@click.command()
@click.option('--dataset', default='wdc', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--title', default=False, help = 'Include Title')
@click.option('--description', default=False, help = 'Include description')
@click.option('--normalized_only', default=True, help = 'Extract only attributes that are viable for normalization')
@click.option('--normalization_params', default="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Binary Classification', 'Name Generalisation','Unit Expansion', 'To Uppercase', 'Delete Marks']", help = 'Which normalizations should be included')
@click.option('--separate', default=False, help = 'Run title and description separately and fuse models after')
def main(dataset, model, verbose, title, description, separate, normalization_params, normalized_only):
    # Load task template
    with open('../prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    normalization_params = ast.literal_eval(normalization_params)

    task_dict['task_name'] = f"zero_shot_normalization_list_{'normalized_only' if normalized_only else ''}"    
    task_dict['task_prefix'] = "Split the product {part} by whitespace. " \
                                "Extract the valid attribute values from the product {part}. Normalize the " \
                                "attribute values according to the guidelines below in JSON format. Unknown attribute values should be marked as 'n/a'. Do not explain your answer. \n" \
                                "Guidelines: \n" \
                                "{guidelines}"
    
    task_dict['dataset_name'] = dataset
    task_dict['data_source'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Normalize the data
    params = "_".join(normalization_params)
    normalized_attributes = normalize_data(split="test", normalization_params=normalization_params)
    file_name_test = f"normalized_test_{params}"
    task_dict = update_task_dict_from_normalized_test_set(task_dict,
                                                          file_name_test,
                                                          title, 
                                                          description, 
                                                          normalized_only, 
                                                          normalized_attributes)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0, request_timeout=120)

    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task_to_json(task_dict['task_name'], task_dict, title, description)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate(messages=[system_setting, human_task_prompt, human_message_prompt])

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    def select_and_run_llm(category, input_text, part='title'):
        guidelines = get_normalization_guidelines_from_csv(normalization_params, category, normalized_only)
        pred = None
        if len(input_text) == 0:
            # No input text provided.
            return pred
        response = chain.run(input=input_text, attributes=', '.join(task_dict['known_attributes'][category]), part=part, guidelines=guidelines)
        print(response)
        try:
            # Convert json string into pydantic model
            pred = pydantic_models[category](**json.loads(response))
            if verbose:
                json_response = json.loads(response)
                print(json_response)
        except ValidationError as valError:
            print(valError)
        except JSONDecodeError as e:
            #converted_response = parse_llm_response_to_json(response)
            cleaned_response_str = response.replace("```json\n", "").replace("\n```", "")
            cleaned_response = json.loads(cleaned_response_str)
            try:
                pred = pydantic_models[category](**cleaned_response)
            except ValidationError as valError:
                print(valError)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
            print('Response: ')
            print(response)
        return pred

    with get_openai_callback() as cb:
        if dataset == 'wdc':
            if not separate:
                if title and description:
                    # append title and description
                    preds = [select_and_run_llm(example['category'], 
                                                f"title: {example['input_title']} \n description: {example['input_description']}", 
                                                part='title and description')
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
                hashed_predictions = {category: {} for category in
                        task_dict['known_attributes']}  # Cache predictions per category
                preds = []

                # Iterate over each product example
                for example in tqdm(task_dict['examples'], disable=not verbose):
                    example_predictions = []

                    # Process the title
                    title_hash = hash(example['input_title'])
                    if title_hash in hashed_predictions[example['category']]:
                        example_predictions.append(hashed_predictions[example['category']][title_hash])
                        if verbose:
                            print("Cached Title")
                    else:
                        title_prediction = select_and_run_llm(example['category'], example['input_title'], part='title')
                        hashed_predictions[example['category']][title_hash] = title_prediction
                        example_predictions.append(title_prediction)

                    # Process the description
                    description_hash = hash(example['input_description'])
                    if description_hash in hashed_predictions[example['category']]:
                        example_predictions.append(hashed_predictions[example['category']][description_hash])
                        if verbose:
                            print("Cached Description")
                    else:
                        description_prediction = select_and_run_llm(example['category'], example['input_description'], part='description')
                        hashed_predictions[example['category']][description_hash] = description_prediction
                        example_predictions.append(description_prediction)

                    # Fuse model predictions
                    final_prediction = fuse_models(pydantic_models[example['category']], *example_predictions)
                    if verbose:
                        print(final_prediction)
                    preds.append(final_prediction)

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