import json
# Update Task Dict
import random
from datetime import datetime
from json import JSONDecodeError

import click
import torch
from dotenv import load_dotenv
from langchain import LLMChain, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pieutils import save_populated_task_to_json, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, parse_llm_response_to_json, subset_task_dict, combine_example
from pieutils.evaluation import evaluate_predictions, calculate_cost_per_1k, visualize_performance, calculate_recall_precision_f1_multiple_attributes
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set, load_known_attribute_values
from pieutils.pydantic_models import ProductCategory
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector


@click.command()
@click.option('--dataset', default='wdc', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='SemanticSimilarity', help='Example selector for few-shot learning')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for in-context learning')
@click.option('--with_containment', default=False, help='Use containment')
@click.option('--with_validation_error_handling', default=False, help='Use validation error handling')
@click.option('--schema_type', default='json_schema', help='Schema to use - json_schema, json_schema_no_type or compact')
@click.option('--no_example_values', default=10, help='Number of example values extracted from training set')
@click.option('--replace_example_values', default=True, help='Replace example values with known attribute values')
@click.option('--title', default=False, help = 'Include Title')
@click.option('--description', default=False, help = 'Include description')
@click.option('--force_from_different_website', default=False, help = 'For WDC Data, ensures that shots come from different Website')
@click.option('--separate', default=False, help = 'Run title and description separately and fuse models after')
def main(dataset, model, verbose, shots, example_selector, train_percentage, with_containment, with_validation_error_handling, schema_type, no_example_values, title, description, force_from_different_website, separate, replace_example_values):
    # Load task template
    with open('../prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "chatgpt_description_with_example_values_in-context_learning_{}_{}_{}_shots_{}_{}_{}-{}_{}_{}_example_values".format(
        dataset, model, shots, example_selector, train_percentage, no_example_values, 'containment_check' if with_containment else '',
        'validation_error_handling' if with_containment else '', schema_type).replace('.', '_').replace('-', '_').replace('/', '_')
    task_dict['task_prefix'] = "Split the product {part} by whitespace. " \
                               "Extract the valid attribute values from the product {part} in JSON format. Keep the " \
                               "exact surface form of all attribute values. All valid attributes are provided in the " \
                               "JSON schema. Unknown attribute values should be marked as 'n/a'. Do not explain your answer."

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dict['shots'] = shots
    task_dict['example_selector'] = example_selector
    task_dict['train_percentage'] = train_percentage

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict,title, description)
    known_attribute_values = load_known_attribute_values(task_dict['dataset_name'], title, description, n_examples=no_example_values,
                                                         train_percentage=train_percentage)
    if verbose:
        print('Known attribute values:')
        print(known_attribute_values)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0)

    default_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0)
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

    # Persist models
    with open('../prompts/meta_models/models_by_{}_{}.json'.format(task_dict['task_name'], 'default_gpt3_5', task_dict['dataset_name']), 'w', encoding='utf-8') as f:
        json.dump(models_json, f, indent=4)

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a world class algorithm for extracting information in structured formats. \n {schema} ")
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])

    # Add few shot sampling
    if example_selector == 'SemanticSimilarity':
        category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(task_dict['dataset_name'], 
                                                                                     list(task_dict['known_attributes'].keys()),
                                                                                     title, description,
                                                                                     load_from_local=False,
                                                                                     k=shots, 
                                                                                     train_percentage=train_percentage, 
                                                                                     force_from_different_website = force_from_different_website)
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

    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate(messages=[system_message_prompt, 
                                          human_task_prompt, 
                                          few_shot_prompt, 
                                          human_task_prompt,
                                          human_message_prompt])
    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    # Containment chain:
    ai_response = AIMessagePromptTemplate.from_template("{response}")
    human_verfification_request = HumanMessagePromptTemplate.from_template(
        'The attribute value(s) "{values}" is/are not found as exact substrings in the product title "{input}". Update all attribute values such that they are substrings of the product title.')
    verification_prompt_list = [system_message_prompt, human_task_prompt, human_message_prompt, ai_response,
                                human_verfification_request]
    verification_prompt = ChatPromptTemplate(messages=verification_prompt_list)
    verification_chain = LLMChain(
        prompt=verification_prompt,
        llm=default_llm,
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

    def validation_error_handling(category, input_text, response, error,  part='title'):

        response_dict = json.loads(response)
        error_attributes = [error['loc'][0] for error in error.errors() if error['type'] == 'type_error.str']

        # Update all attributes that are not strings to strings by LLM
        pred = None
        error_handling_response = error_handling_chain.run(input=input_text,
                                                           schema=convert_to_json_schema(pydantic_models[category]),
                                                           response=json.dumps(response_dict), error="', '".join(error_attributes), part=part)
        try:
            # Convert json string into pydantic model
            print(json.loads(error_handling_response))
            pred = pydantic_models[category](**json.loads(error_handling_response))
        except ValidationError as valError:
            pred = validation_error_handling(category, input_text, response, valError)
        except JSONDecodeError as e:
            print('Error: {}'.format(e))

        return pred, error_handling_response

    def select_and_run_llm(category, input_text, part='title', with_validation_error_handling=True,
                           with_containment=True, schema_type='json_schema', url=None):
        pred = None
        try:
            response = chain.run(input=input_text,
                                 schema=convert_to_json_schema(pydantic_models[category], schema_type=schema_type),
                                 part=part, category=category, url = url)

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

            if pred is not None and with_containment:
                # Verify if the extracted attribute values are contained in the input text and make sure that no loop is entered.
                not_found_values = [v for v in pred.dict().values() if
                                    v is not None and v != 'n/a' and v not in input_text]
                previous_responses = []
                while len(not_found_values) > 0 and response not in previous_responses:
                    # Verify extracted attribute values
                    if verbose:
                        print('Not found values: {}'.format(not_found_values))
                    # Try this only once to avoid infinite loops
                    verified_response = verification_chain.run(input=input_text,
                                                               schema=convert_to_json_schema(pydantic_models[category]),
                                                               response=response, values='", "'.join(not_found_values),
                                                               part=part)
                    previous_responses.append(response)
                    try:
                        if verbose:
                            print(json.loads(verified_response))
                        pred = pydantic_models[category](**json.loads(verified_response))
                        not_found_values = [v for v in pred.dict().values() if
                                            v is not None and v != 'n/a' and v not in input_text]
                    except ValidationError as valError:
                        if with_validation_error_handling:
                            pred, verified_response = validation_error_handling(category, input_text, verified_response,
                                                                                valError, part)
                        else:
                            print('Validation Error: {}'.format(valError))
                    except JSONDecodeError as e:
                        print('Error: {}'.format(e))
        except Exception as e:
            print(e)
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

    if dataset == "wdc":
        # Evaluate performance only for normalizable attributes
        task_dict_subset = subset_task_dict(task_dict)

        targets = [example['target_scores'] for example in task_dict_subset['examples']]
        preds = [example['pred'] for example in task_dict_subset['examples']]
        categories = [example['category'] for example in task_dict_subset['examples']]
        postprocessed_preds = [pred if pred is not None else '' for pred in preds]

        task_dict_subset['examples'] = [combine_example(example, pred, post_pred)
                                for example, pred, post_pred in
                                zip(task_dict_subset['examples'], postprocessed_preds, postprocessed_preds)]

        task_dict_subset['results'] = calculate_recall_precision_f1_multiple_attributes(targets, postprocessed_preds, categories,
                                                                                task_dict_subset['known_attributes'])

        #print(task_dict['results'])
        print("Performance on attributes which are considered in normalization:")
        print(task_dict_subset['results']['micro'])

        task_dict_subset['task_name'] = f"{task_dict['task_name']}_subset"
        save_populated_task_to_json(task_dict_subset['task_name'], task_dict, title, description)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()