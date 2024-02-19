import json
# Update Task Dict
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
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import save_populated_task_to_json, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, parse_llm_response_to_json, get_normalization_guidelines_from_csv
from pieutils.evaluation import evaluate_predictions, calculate_cost_per_1k, visualize_performance
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_normalized_test_set, load_known_value_correspondences_for_normalized_attributes
from pieutils.normalization import normalize_data
from pieutils.pydantic_models import ProductCategory


@click.command()
@click.option('--dataset', default='wdc', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--with_validation_error_handling', default=False, help='Use validation error handling')
@click.option('--schema_type', default='json_schema', help='Schema to use - json_schema, json_schema_no_type or compact')
@click.option('--replace_example_values', default=True, help='Replace example values with known attribute values')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for example value selection')
@click.option('--no_example_values', default=3, help='Number of example values extracted from training set')
@click.option('--title', default=False, help = 'Include Title')
@click.option('--description', default=False, help = 'Include description')
@click.option('--normalization_params', default="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Binary Classification', 'Name Generalisation','Unit Expansion', 'To Uppercase', 'Delete Marks']", help = 'Which normalizations should be included')
@click.option('--normalized_only', default=True, help = 'Extract only attributes that are viable for normalization')
def main(dataset, model, verbose, with_validation_error_handling, schema_type, replace_example_values, train_percentage, no_example_values, title, description,normalization_params, normalized_only):
    # Load task template
    with open('../prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    normalization_params = ast.literal_eval(normalization_params)

    task_dict['task_name'] = f"zero_shot_normalization_{no_example_values}_examples_with_correspondence_{schema_type}_{'normalized_only' if normalized_only else ''}"
    task_dict['task_prefix'] = "Split the product {part} by whitespace. " \
                                "Extract the valid attribute values from the product {part}. Normalize the " \
                                "attribute values according to the guidelines below in JSON format. Unknown attribute values should be marked as 'n/a'. Do not explain your answer. \n" \
                                "Guidelines: \n" \
                                "{guidelines}"

    task_dict['data_source'] = dataset
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Normalize the data
    train_split = f"train_{train_percentage}"
    normalized_attributes = normalize_data(split=train_split, normalization_params=normalization_params)
    normalized_attributes = normalize_data(split="test", normalization_params=normalization_params)

    # Make normalized data a task_dict
    params = "_".join(normalization_params)
    file_name_test = f"normalized_test_{params}"
    file_name_train= f"normalized_{train_split}_{params}"
    task_dict = update_task_dict_from_normalized_test_set(task_dict,file_name_test,title, description, normalized_only, normalized_attributes)

    # Load example values (as specified in the product texts)
    known_attribute_values = load_known_value_correspondences_for_normalized_attributes(task_dict['dataset_name'], 
                                                                                    normalized_only, 
                                                                                    normalized_attributes,
                                                                                    file_name_train=file_name_train,
                                                                                    n_examples=no_example_values,
                                                                                    train_percentage=train_percentage)

    if verbose:
        print('Known attribute values:')
        print(known_attribute_values)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0, request_timeout=120)

    default_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0)

    # Save populated task and make sure that it is saved in the correct folder!
    save_populated_task_to_json(task_dict['task_name'], task_dict, title, description)

    # Ask LLM to generate meta models for each product category and attribute

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a world class algorithm for creating "
                                                                      "descriptions of product categories and their "
                                                                      "attributes following this JSON schema: \n {schema}.")
    human_task_meta_model = HumanMessagePromptTemplate.from_template("Write short descriptions for the product category"
                                                                      " {category} and the attributes {attributes} that are helpful to identify relevant attribute values in product titles."
                                                                     "The descriptions should not be longer than one sentence."
                                                                      "The following attribute values are known for each attribute: \n"
                                                                     "{known_attribute_values}. \n" 
                                                                     "Respond with a JSON object following the provided schema.")
    prompt_meta_model = ChatPromptTemplate(messages=[system_message_prompt, human_task_meta_model])
    
    pydantic_models = {}
    models_json = {}
    for category in task_dict['known_attributes']:
        print('Create model for category: {}'.format(category))

        chain = LLMChain(
            prompt=prompt_meta_model,
            llm=default_llm,
            verbose=verbose
        )

        #chain_meta_model = create_structured_output_chain(ProductCategory, llm, prompt_meta_model, verbose=True)
        known_attribute_values_per_category = json.dumps(known_attribute_values[category])
        
        response = chain.run({'schema': convert_to_json_schema(ProductCategory, False), 
                              'category': category,
                              'attributes': ', '.join(task_dict['known_attributes'][category]),
                              'known_attribute_values': known_attribute_values_per_category})
        try:
            print(response)
            pred = ProductCategory(**json.loads(response))
            print(pred)
        except JSONDecodeError as e:
            print('JSONDecoder Error: {}'.format(e))
            print('Response: {}'.format(response))
            continue
        except ValidationError as e:
            print('Validation Error: {}'.format(e))
            print('Response: {}'.format(response))
            # Most likely the model did not generate any examples
            response_dict = json.loads(response)
            for attribute in response_dict['attributes']:
                if 'examples' not in attribute:
                    attribute['examples'] = []
            pred = ProductCategory(**response_dict)

        if replace_example_values:
            pydantic_models[category] = create_pydanctic_model_from_pydantic_meta_model(pred, known_attribute_values[category])
        else:
            pydantic_models[category] = create_pydanctic_model_from_pydantic_meta_model(pred)

        models_json[category] = create_dict_of_pydanctic_product(pydantic_models[category])


    updated_schemas = {category: convert_to_json_schema(model, False) for category, model in pydantic_models.items()}
    

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a world class algorithm for extracting information in structured formats. \n {schema}")
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt_list = [system_message_prompt, human_task_prompt, human_message_prompt]
    prompt = ChatPromptTemplate(messages=prompt_list)

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    # Error handling chain:
    ai_response = AIMessagePromptTemplate.from_template("{response}")
    human_error_handling_request = HumanMessagePromptTemplate.from_template("Change the attributes '{error}' to string values. List values should be concatenated by ' '.")
    error_handling_prompt_list = [system_message_prompt, human_task_prompt, human_message_prompt, ai_response, human_error_handling_request]
    error_handling_prompt = ChatPromptTemplate(messages=error_handling_prompt_list)
    error_handling_chain = LLMChain(
        prompt=error_handling_prompt,
        llm=default_llm,
        verbose=verbose
    )

    def validation_error_handling(category, input_text, response, error, part='title'):

        response_dict = json.loads(response)
        error_attributes = [error['loc'][0] for error in error.errors() if error['type'] == 'type_error.str']

        # Update all attributes that are not strings to strings by LLM
        pred = None
        error_handling_response = error_handling_chain.run(input=input_text,
                                                        schema=convert_to_json_schema(pydantic_models[category]),
                                                        response=json.dumps(response_dict), error="', '".join(error_attributes), part=part)
        try:
            # Convert json string into pydantic model
            if verbose:
                print(json.loads(error_handling_response))
            pred = pydantic_models[category](**json.loads(error_handling_response))
        except ValidationError as valError:
            pred = validation_error_handling(category, input_text, response, valError)
        except JSONDecodeError as e:
            print('Error: {}'.format(e))

        return pred, error_handling_response

    def select_and_run_llm(category, input_text, part='title', updated_schemas=updated_schemas, with_validation_error_handling=with_validation_error_handling, schema_type='json_schema'):
        pred = None

        guidelines = get_normalization_guidelines_from_csv(normalization_params, category, normalized_only)

        try:
            updated_schema = updated_schemas[category]
            response = chain.run(input=input_text, schema=updated_schema, part=part, guidelines=guidelines)

            try:
                # Convert json string into pydantic model
                if verbose:
                    print(json.loads(response))
                pred = pydantic_models[category](**json.loads(response))
            except ValidationError as valError:
                if with_validation_error_handling:
                    pred, response = validation_error_handling(category, input_text, response, valError)
                else:
                    print('Validation Error: {}'.format(valError))
            except JSONDecodeError as e:
                converted_response = parse_llm_response_to_json(response)
                if verbose:
                    print(converted_response)
                try:
                    pred = pydantic_models[category](**converted_response)
                except ValidationError as valError:
                    if with_validation_error_handling:
                        pred, response = validation_error_handling(category, input_text, response, valError)
                    else:
                        print('Validation Error: {}'.format(valError))
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)
                print('Response: ')
                print(response)

        except Exception as e:
            print(f"Exception: {e}")
        return pred

    # Run LLM
    with get_openai_callback() as cb:
        if dataset == 'wdc':
            if title and description:
                # append title and description
                preds = [select_and_run_llm(example['category'], 
                                            updated_schemas=updated_schemas,
                                            input_text = f"title: {example['input_title']} \n description: {example['input_description']}", 
                                            part='title and description')
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]
            elif title and not description:
                preds = [select_and_run_llm(example['category'], 
                                            updated_schemas=updated_schemas,
                                            input_text = f"title: {example['input_title']}", 
                                            part='title')
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]
            elif description and not title:
                preds = [select_and_run_llm(example['category'], 
                                            updated_schemas=updated_schemas,
                                            input_text = f"description: {example['input_description']}", 
                                            part='description')
                                            for example in tqdm(task_dict['examples'], disable=not verbose)]

        else:
            preds = [select_and_run_llm(example['category'],
                                        updated_schemas=updated_schemas,
                                        input_text = example['input']) for example in
                    tqdm(task_dict['examples'], disable=not verbose)]

        task_dict['total_tokens'] = cb.total_tokens
        task_dict['prompt_tokens'] = cb.prompt_tokens
        task_dict['completion_tokens'] = cb.completion_tokens
        task_dict['total_costs'] = cb.total_cost
        total_costs=cb.total_cost

        print(f"Total Tokens: {task_dict['total_tokens']}")
        print(f"Total Cost: {task_dict['total_costs']}")

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
