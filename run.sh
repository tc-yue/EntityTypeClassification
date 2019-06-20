#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python -u main.py --do_train --cuda \
    --do_valid \
    --do_test \
    --train_data_path processed/train_figer.pkl \
    --valid_data_path processed/test_figer.pkl \
    --attention_mode bilinear \
    --embedding_data_path processed/embedding_matrix.pkl \
    --embedding \
    -lr 0.001 -e 20 \
    -save models/lstm