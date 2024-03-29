import json
import random
from datetime import datetime
from json import JSONDecodeError

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

from pieutils import create_pydanctic_models_from_known_attributes, parse_llm_response_to_json, save_populated_task_to_json, subset_task_dict
from pieutils.evaluation import evaluate_predictions, visualize_performance, calculate_cost_per_1k, combine_example, calculate_recall_precision_f1_multiple_attributes
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set
from pieutils.search import CategoryAwareSemanticSimilarityExampleSelector


@click.command()
@click.option('--dataset', default='oa-mine', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=True, help='Verbose mode')
@click.option('--shots', default=3, help='Number of shots for few-shot learning')
@click.option('--example_selector', default='MaxMarginalRelevance', help='Example selector for few-shot learning')
@click.option('--train_percentage', default=1.0, help='Percentage of training data used for in-context learning')
@click.option('--title', default=False, help = 'Include Title')
@click.option('--description', default=False, help = 'Include description')
@click.option('--force_from_different_website', default=False, help = 'For WDC Data, ensures that shots come from different Website')
@click.option('--separate', default=False, help = 'Run title and description separately and fuse models after')
def main(dataset, model, verbose, shots, example_selector, train_percentage, title, description, force_from_different_website, separate):
    # Load task template
    with open('../prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_name'] = "{}_multiple_attribute_values-great-incontext-{}-{}-shots_train_perc_{}_examples_by_{}_{}".format('title_and_description' if title and description else ('title' if title else "description"),dataset, shots, train_percentage, example_selector, 'force_from_different_website'if force_from_different_website else '')
    task_dict['task_prefix'] = "Extract the attribute values from the product {part} below in a JSON format. Valid " \
                                "attributes are {attributes}. If an attribute is not present in the product title, " \
                                "the attribute value is supposed to be 'n/a'. Do not explain your answer."

    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_dict['train_percentage'] = train_percentage

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict,title, description)

    # Initialize model
    task_dict['model'] = model
    if 'gpt-3' in task_dict['model'] or 'gpt-4' in task_dict['model']:
        llm = ChatOpenAI(model_name=task_dict['model'], temperature=0, request_timeout=120)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")

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

    prompt = ChatPromptTemplate(messages=[system_setting, human_task_prompt, few_shot_prompt,
                                          human_task_prompt, human_message_prompt])

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )

    def select_and_run_llm(category, input_text, part='title', url=None):
        pred = None
        try:
            response = chain.run(input=input_text, attributes=', '.join(task_dict['known_attributes'][category]),
                             category=category, part=part, url=url)
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

        print("Performance on attributes which are considered in normalization:")
        print(task_dict_subset['results']['micro'])

        task_dict_subset['task_name'] = f"{task_dict['task_name']}_subset"
        save_populated_task_to_json(task_dict_subset['task_name'], task_dict, title, description)

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
