import random

import click
from dotenv import load_dotenv

from pieutils.preprocessing import convert_to_mave_dataset, reduce_training_set_size, convert_to_open_tag_format, preprocess_wdc


@click.command()
@click.option('--dataset', default='wdc', help='Dataset name')
def main(dataset):
    """Preprocess datasets."""
    if dataset == "wdc":
        preprocess_wdc()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    for percentage in [0.2, 1.0]:
        if dataset != "wdc":
            reduce_training_set_size(dataset, percentage=percentage)
        convert_to_mave_dataset(dataset, percentage=percentage, skip_test=False)
        #convert_to_open_tag_format(dataset, percentage=percentage, skip_test=False)


if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
