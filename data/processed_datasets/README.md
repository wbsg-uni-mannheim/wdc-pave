# Benchmark datasets

We provide the following benchmark datasets for the evaluation of attribute value extraction and normalization:

- **wdc**: Contains product offers and product attribute-value pairs for attributes that require normalization. The attribute values are *not* normalized.
- **wdc_normalized**: Contains product offers and product attribute-value pairs for attributes that require normalization. The attribute values are normalized.
- **wdc_with_all_attributes**: Contains product offers and product attribute-value pairs for all annotated attributes. The attribute values are *not* normalized.
- **wdc_with_all_attributes_normalized**: Contains product offers and product attribute-value pairs for all annotated attributes. The attribute values are normalized.

The datasets are stored in the `data/processed_datasets` folder. The datasets are stored in the following format:
- **train.json**: Small training set.
- **train_large.json**: Large training set.
- **test.json**: Test set.

In our [paper](https://arxiv.org/pdf/2403.02130), we use the `wdc` dataset for the evaluation of attribute value extraction and `wdc_normalized` dataset for the evaluation of attribute value extraction the normalization. 
Results for the datasets `wdc_with_all_attributes` and `wdc_with_all_attributes_normalized` datasets are not reported in the paper but can be used for further evaluation.