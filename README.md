# Using LLMs for the Extraction and Normalization of Product Attribute Values
This repository contains code and data for experiments on attribute value extraction and normalization using large language models.

## Requirements

We evaluate hosted LLMs, such as GPT-3.5 and GPT-4.
Therefore, an OpenAI access tokens needs to be placed in a `.env` file at the root of the repository.
To obtain this OpenAI access token, users must [sign up](https://platform.openai.com/signup) for an OpenAI account.

## Installation
The codebase requires python 3.9 To install dependencies we suggest to use a conda virtual environment:

```
conda create -n wdc-pave python=3.9
conda activate wdc-pave
pip install -r requirements.txt
pip install pieutils/
```

## Dataset
The extraction and extraction&normalization WDC PAVE data can be found in the data/processed_datasets folder.

## Prompts
We experiment with various prompt templates involving descriptions and example values, and adding demonstrations. The following figure shows the prompt structures for the two schema descriptions (black font for extraction, black + red font for extraction + normalization).

![Prompt Designs](resources/prompt_template.PNG)

### Execution
The prompts and the code to execute the prompts are defined in the folder `prompts`.
You can run the prompts with the following scripts:

```
scripts/01_run_example_values_prompts.sh
scripts/02_run_prompts_with_training_data.sh
scripts/08_run_prompts_for_data_normalization.sh
```
