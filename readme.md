
# Surprising Event Boundaries in Narratives

This project seeks to predict surprising event boundaries in stories.

## Dependencies

```sh
pip install -r requirements.txt
```
Python3 is required

GPU (>= 16GB memory) is highly recommended.

## Feature Extraction

```sh
sh extract_all_features.sh
```
To run this source code, several external resources are required to be downloaded separately. To facilitate understanding of the features without having to obtain each of them, we have extracted all features under ```data/```

External resources required:

1. Event annotated data (See Sap et al., 2021)
2. Story Cloze Test dataset (See Mostafazadeh  et  al., 2016)
3. Atomic relation dataset (See Sap et al., 2019)
4. Glucose relation dataset (See Mostafazadeh  et  al., 2020)
5. Realis-annotated dataset (See Sims et al., 2019)
6. URL and Access Key for using Turing-NLG model (See Rosset, 2020)

## Model Training

To train on detecting event boundaries:

```sh
sh train.sh event_boundaries
```

To train on identifying commonsense and nonsense story endings

```sh
sh train.sh story_cloze
```

To predict event boundaries on the story cloze dataset

```sh
python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_30_1 --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_confidence --fold_number 1 --inference_only True --load_trained_model True --infer_story_cloze True --story_cloze_train_dataset_filename data/all_features_including_annotations_story_cloze_with_prior_confidence.csv

```

## Analysis

Please run after model training. ```ROOT_DIR``` refers to the root folder that contains all of trained model config folders.


To plot feature weights that support understanding of informative features:

```sh
python interpret_features.py --root_dir ROOT_DIR
```

To conduct significance testing using McNemar's test

```sh
python mcnemar_test.py --root_dir ROOT_DIR
```


To correlate story ending with predicted event boundaries:

```sh
python correlate_story_cloze_to_predicted_surprise.py --filename ROOT_DIR/surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_30_1/surprise_discriminator_eval_tokens_epoch_.csv
```
