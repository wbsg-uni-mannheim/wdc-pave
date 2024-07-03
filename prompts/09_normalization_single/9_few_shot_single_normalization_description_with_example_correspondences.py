import ast
import json
# Update Task Dict
import random
from datetime import datetime
from json import JSONDecodeError

import click
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import ValidationError
from tqdm import tqdm

from pieutils import save_populated_task_to_json, create_pydanctic_model_from_pydantic_meta_model, \
    create_dict_of_pydanctic_product, convert_to_json_schema, parse_llm_response_to_json, \
    get_normalization_guidelines_from_csv
from pieutils.evaluation import evaluate_predictions, calculate_cost_per_1k, visualize_performance
from pieutils.fusion import fuse_models
from pieutils.normalization import normalize_data
from pieutils.preprocessing import update_task_dict_from_normalized_test_set, \
    load_known_value_correspondences_for_normalized_attributes, update_task_dict_from_test_set
from pieutils.pydantic_models import ProductCategory
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector


@click.command()
@click.option('--dataset', default='wdc', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-16k-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--shots', default=10, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='SemanticSimilarity', help='Example selector for few-shot learning')
@click.option('--train_percentage', default=0.2, help='Percentage of training data used for in-context learning')
@click.option('--with_validation_error_handling', default=False, help='Use validation error handling')
@click.option('--schema_type', default='json_schema', help='Schema to use - json_schema, json_schema_no_type or compact')
@click.option('--no_example_values', default=10, help='Number of example values extracted from training set')
@click.option('--title', default=True, help = 'Include Title')
@click.option('--replace_example_values', default=True, help='Replace example values with known attribute values')
@click.option('--description', default=True, help = 'Include description')
@click.option('--force_from_different_website', default=False, help = 'For WDC Data, ensures that shots come from different Website')
@click.option('--normalization_params', default="['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Binary Classification', 'Name Generalisation','Unit Expansion', 'To Uppercase', 'Delete Marks']", help = 'Which normalizations should be included')
@click.option('--separate', default=False, help = 'Run title and description separately and fuse models after')
@click.option('--normalized_only', default=True, help = 'Extract only attributes that are viable for normalization')
def main(dataset, model, verbose, shots, example_selector, train_percentage, with_validation_error_handling, schema_type, no_example_values, title, description, force_from_different_website, separate, normalized_only, normalization_params, replace_example_values):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    # Load second task template for raw-extracted attribute values
    with open('prompts/task_template.json', 'r') as f:
        task_dict_raw_attribute_values = json.load(f)

    normalization_params = ast.literal_eval(normalization_params)
    normalized_dataset_name = f"{dataset}_normalized"

    task_dict['task_name'] = "description_with_example_values_single_normalization_in-context_learning_{}_{}_{}_shots_{}_{}-{}_example_correspondences_{}_{}".format(
        dataset, model, shots, example_selector, train_percentage,no_example_values,
        'validation_error_handling' if with_validation_error_handling else '', schema_type).replace('.', '_').replace('-', '_').replace('/', '_')
    
    task_dict['task_prefix'] =  "The provided attribute value has been extracted from the mentioned product title and description. \n" \
                                "Normalize the attribute value according to the guidelines below in JSON format. \n"\
                                "Unknown attribute values should be marked as 'n/a'. " \
                                "Do not explain your answer. \n" \
                                "Guidelines: \n" \
                                "{guidelines}"

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dict['shots'] = shots
    task_dict['example_selector'] = example_selector
    task_dict['train_percentage'] = train_percentage

    task_dict_raw_attribute_values['dataset_name'] = dataset
    task_dict_raw_attribute_values['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_split = f"train_{train_percentage}"
    normalized_attributes = normalize_data(split=train_split, normalization_params=normalization_params)
    normalized_attributes = normalize_data(split="test", normalization_params=normalization_params)

    # Load examples into task dict
    params = "_".join(normalization_params)
    file_name_test = f"normalized_test_{params}"
    file_name_train= f"normalized_{train_split}_{params}"
    task_dict = update_task_dict_from_normalized_test_set(task_dict,file_name_test,title, description, normalized_only, normalized_attributes)

    task_dict_raw_attribute_values = update_task_dict_from_test_set(task_dict_raw_attribute_values, title, description)

    # Add goldstandard not normalized example values to task_dict
    for example, raw_example in zip(task_dict['examples'], task_dict_raw_attribute_values['examples']):
        # Convert target scores to json
        attributes_for_normalisation = normalized_attributes[example['category']]
        raw_attribute_values = {attribute: list(value.keys())[0] for attribute, value in
                                raw_example['target_scores'].items() if attribute in attributes_for_normalisation}
        example['input_extracted_attribute_values'] = json.dumps(raw_attribute_values, indent=4)

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
    with open('prompts/meta_models/models_by_{}_{}.json'.format(task_dict['task_name'], 'default_gpt3_5', task_dict['dataset_name']), 'w', encoding='utf-8') as f:
        json.dump(models_json, f, indent=4)

    updated_schemas = {category: convert_to_json_schema(model, False) for category, model in pydantic_models.items()}

    # Create Chains
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a world class algorithm for normalizing structured attribute-value pairs. \n {schema}")
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])

    # Add few shot sampling
    if example_selector == 'SemanticSimilarity':
        category_example_selector = CategoryAwareSemanticSimilarityExampleSelector(normalized_dataset_name, 
                                                                                     list(task_dict['known_attributes'].keys()),
                                                                                     title, description,
                                                                                     load_from_local=False, 
                                                                                     k=shots, train_percentage=train_percentage, 
                                                                                     normalization_params=normalization_params, 
                                                                                     normalized_only=normalized_only, 
                                                                                     normalized_attributes=normalized_attributes, 
                                                                                     force_from_different_website = force_from_different_website,
                                                                                     add_raw_extractions=True)
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

    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}\n Attribute-Value Pair: {pair}")
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

    def select_and_run_llm(category, input_text, example, updated_schemas=updated_schemas, part='title', with_validation_error_handling=True, schema_type='json_schema', url=None):
        pred = None
        guidelines = get_normalization_guidelines_from_csv(normalization_params, category, normalized_only)
        relevant_attributes = [attribute for attribute in updated_schemas[category]['parameters']['properties'].keys()]

        complete_pred = {}
        for i in range(0, len(relevant_attributes)):
            # Filter for relevant attribute
            attribute = relevant_attributes[i]
            updated_guidelines = []
            find_next_attribute = False

            if attribute in ["Width", "Height", "Depth", "Length"]:
                searched_string = "Width/Height/Depth/Length"
                other_attributes = [other_attribute for other_attribute in relevant_attributes
                                    if other_attribute not in ["Width", "Height", "Depth", "Length"]]
            else:
                searched_string = attribute
                other_attributes = [other_attribute for other_attribute in relevant_attributes
                                    if other_attribute != attribute]

            for part in guidelines.split('\n'):
                if find_next_attribute:
                    if any([part.startswith(other_attribute) for other_attribute in other_attributes]):
                        break
                    else:
                        updated_guidelines.append(part)
                        continue

                if part.startswith(searched_string):
                    updated_guidelines.append(part)
                    find_next_attribute = True

            updated_guidelines = '\n'.join(updated_guidelines)

            # Update Schema
            updated_schema = {'name': updated_schemas[category]['name'],
                              'description': updated_schemas[category]['description'],
                              'parameters': {'type': 'object',
                                             'properties': {
                                                 attribute: updated_schemas[category]['parameters']['properties'][
                                                     attribute]}}}

            # Get extracted attribute value
            pair = json.dumps({attribute: json.loads(example['input_extracted_attribute_values'])[attribute]})

            # Change relevant attributes in SemanticSimilaritySelector
            category_example_selector.attributes = [[attribute]]

            try:
                response = chain.run(input=input_text, schema=updated_schema, part=part, pair=pair,
                                     category=category, guidelines=updated_guidelines, url=url)

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
                print(e)

            pred_dict = pred.dict()
            complete_pred[attribute] = pred_dict[attribute]
        try:
            complete_pred = pydantic_models[category](**complete_pred)
        except Exception as e:
            print(e)

        return complete_pred

    with get_openai_callback() as cb:
        if dataset == 'wdc':
            if not separate:
                if title and description:
                    # append title and description
                    preds = [select_and_run_llm(example['category'],
                                                updated_schemas=updated_schemas,
                                                input_text=f"title: {example['input_title']} \n" \
                                                           f"description: {example['input_description']}",
                                                example=example,
                                                part='title and description')
                             for example in tqdm(task_dict['examples'], disable=not verbose)]
                elif title and not description:
                    preds = [select_and_run_llm(example['category'],
                                                updated_schemas=updated_schemas,
                                                input_text=f"title: {example['input_title']}",
                                                example=example,
                                                part='title')
                             for example in tqdm(task_dict['examples'], disable=not verbose)]
                elif description and not title:
                    preds = [select_and_run_llm(example['category'],
                                                updated_schemas=updated_schemas,
                                                input_text=f"description: {example['input_description']}",
                                                example=example,
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

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()