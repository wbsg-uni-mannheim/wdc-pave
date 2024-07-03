import json
import os

# Iterate over all the files in the results directory and collect the results
path = 'prompts/runs/normalisation'
for filename in os.listdir(path):
    with open(os.path.join(path, filename), 'r') as file:
        task_dict = json.loads(file.read())

        if 'results' not in task_dict:
            continue
        line = f'{task_dict["task_name"]} \t'
        line += f'{task_dict["model"]} \t'
        line += f'{task_dict["results"]["micro"]["micro_precision"]:.2f} \t'
        line += f'{task_dict["results"]["micro"]["micro_recall"]:.2f} \t'
        line += f'{task_dict["results"]["micro"]["micro_f1"]:.2f} \t'

        # Add token information if available
        line += f'{task_dict["prompt_tokens"]} \t'
        line += f'{task_dict["completion_tokens"]} \t'
        line += f'{task_dict["total_tokens"]} \t'

        print(line)