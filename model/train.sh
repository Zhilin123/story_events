#!/bin/sh
if [ $1 == 'event_boundaries' ]
then
  for i in {1..10}
  do
    #event boundary detector with roberta
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_30_{$i} --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_confidence --fold_number $i
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_1_{$i} --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 1 --feature_mode prior_confidence --fold_number $i
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_features_limit_30_{$i} --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_features --fold_number $i

    #roberta only
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_only_roberta_{$i} --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --only_roberta True --fold_number $i

    #event boundary detector without roberta
    python surprise_ranker_model.py --config_name surprise_ranker_lr_1e-3_gru_new_prior_confidence_limit_30_{$i}_gru_only --lr 1e-3 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_confidence --fold_number $i  --gru_only True
    python surprise_ranker_model.py --config_name surprise_ranker_lr_1e-3_gru_new_prior_confidence_limit_1_{$i}_gru_only --lr 1e-3 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 1 --feature_mode prior_confidence --fold_number $i  --gru_only True
    python surprise_ranker_model.py --config_name surprise_ranker_lr_1e-3_gru_new_prior_features_limit_30_{$i}_gru_only --lr 1e-3 --train_dataset_filename data/all_features_including_annotations_prev_sent_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_features --fold_number $i --gru_only True

else
  for i in {1..10}
  do
    #event boundary detector with roberta
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_30_fold_{$i}_story_cloze --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_story_cloze_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_confidence --fold_number $i --story_cloze True
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_confidence_limit_1_fold_{$i}_story_cloze --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_story_cloze_with_prior_confidence.csv --gru True --gru_length_limit 1 --feature_mode prior_confidence --fold_number $i --story_cloze True
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_new_prior_features_limit_30_fold_{$i}_story_cloze --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_story_cloze_with_prior_confidence.csv --gru True --gru_length_limit 30 --feature_mode prior_features --fold_number $i --story_cloze True

    #roberta only
    python surprise_ranker_model.py --config_name surprise_ranker_lr_5e-6_gru_only_roberta_fold_{$i}_story_cloze --lr 5e-6 --train_dataset_filename data/all_features_including_annotations_story_cloze_with_prior_confidence.csv --only_roberta True --fold_number $i --story_cloze True
  done
