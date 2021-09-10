#!/bin/sh

# atomic
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_neg_sample --train_dataset_filename v4_atomic_all_agg.csv --add_negative_samples True --glucose False

python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_neg_sample --train_dataset_filename v4_atomic_all_agg.csv --custom_dataset_filename hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse True --custom_save_filename custom_all_confidence_reverse.csv
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_neg_sample --train_dataset_filename v4_atomic_all_agg.csv --custom_dataset_filename hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse False --custom_save_filename custom_all_confidence.csv

python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_neg_sample --train_dataset_filename v4_atomic_all_agg.csv --custom_dataset_filename cloze_test_val_winter2018.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse True --custom_save_filename custom_all_confidence_reverse_story_cloze.csv
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_neg_sample --train_dataset_filename v4_atomic_all_agg.csv --custom_dataset_filename cloze_test_val_winter2018.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse False --custom_save_filename custom_all_confidence_story_cloze.csv

#glucose
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_glucoseNL_neg_sample --train_dataset_filename GLUCOSE_training_data_final.csv --add_negative_samples True --glucose True

python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_glucoseNL_neg_sample --train_dataset_filename GLUCOSE_training_data_final.csv --custom_dataset_filename hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse True --glucose True --custom_save_filename glucose_all_confidence_reverse.csv
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_glucoseNL_neg_sample --train_dataset_filename GLUCOSE_training_data_final.csv --custom_dataset_filename hippocorpus_paragraph_type_and_surprise_annotation_by_sentence_with_extracted.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse False --glucose True --custom_save_filename glucose_all_confidence.csv

python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_glucoseNL_neg_sample --train_dataset_filename GLUCOSE_training_data_final.csv --custom_dataset_filename cloze_test_val_winter2018.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse True --glucose True --custom_save_filename glucose_all_confidence_reverse_story_cloze.csv
python atomic_and_glucose_model.py --config_name model_batch_size64_roberta_glucoseNL_neg_sample --train_dataset_filename GLUCOSE_training_data_final.csv --custom_dataset_filename cloze_test_val_winter2018.csv --add_negative_samples True --generate_custom True --inference_only True  --load_trained_model True --hippocorpus_category all --structured False --hippo_reverse False --glucose True --custom_save_filename glucose_all_confidence_story_cloze.csv

#realis
python trainTagger.py
python bertTagger.py --eval_data realis_hippocorpus_input_all.csv --pred_output_file realis_hippocorpus_output_all.pkl --do_eval --output_dir realisTagger_e3_r1 --txtCol sents
python bertTagger.py --eval_data cloze_test_val_winter2018_realis_input.csv --pred_output_file cloze_test_val_winter2018_realis_input.pkl --do_eval --output_dir realisTagger_e3_r1 --txtCol sents

#sequentiality
python extractLinearity.py --input_story_file hippocorpus_extract_linearity_input.csv --history_sizes -1 0 1 --output_sentence_file hippocorpus.gpt2Pplx.-1.0.1.sentenceLevel.csv --output_story_file hippocorpus.gpt2Pplx.-1.0.1.pkl --language_model gpt2 --device cuda --story_id_column AssignmentId
python extractLinearity.py --input_story_file cloze_test_val_winter2018_extract_linearity_input.csv --history_sizes -1 0 1 --output_sentence_file storycloze.gpt2Pplx.-1.0.1.sentenceLevel.csv --output_story_file storycloze.gpt2Pplx.-1.0.1.pkl --language_model gpt2 --device cuda --story_id_column AssignmentId

#SimGen
python call_tnlg.py --mode surprise --tnlg_access_url_and_access_key xxx
python get_cosine_similarity_tnlg_and_ground_sentence.py --src surprise

python call_tnlg.py --mode story_cloze --tnlg_access_url_and_access_key xxx
python get_cosine_similarity_tnlg_and_ground_sentence.py --src story_cloze

#Combine features
python combine_features.py --mode prev_sent
python combine_features.py --mode story_cloze
