#!/bin/bash

python3 step1_generate_word_vector.py

python3 step2_process_data_format_by_keras.py

python3 step3_train_cnn_model.py

python3 step4_use_cnn_model_word_segmentation.py

python3 step5_check_final_result.py

./data/icwb2-data/scripts/score data/icwb2-data/gold/msr_training_words.utf8 data/icwb2-data/gold/msr_test_gold.utf8 data/icwb2-data/testing/msr_test.split.tag2word.utf8 > deep.score

