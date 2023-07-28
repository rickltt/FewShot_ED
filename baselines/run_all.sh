#!/bin/bash

DATA_DIR='../data/ere'
OUTPUT_DIR='./ere_output'

MODEL=(cosine euclidean dot relation)
CRF=(pacrf vanilla)
N=(5 10)
K=(5 10)
for m in ${MODEL[@]}
do
    for n in ${N[@]}
    do
        for k in ${K[@]}
        do
            CUDA_VISIBLE_DEVICES='5' python main.py \
                --data_dir $DATA_DIR \
                --do_train \
                --output_dir $OUTPUT_DIR \
                --max_len 128 \
                --learning_rate 2e-5 \
                --model unified \
                --metric $m \
                --N $n \
                --K $k \
                --train_epoch 10000 \
                --eval_epoch 500 \
                --eval_step 500 \
                --test_epoch 3000
            for i in {1..5}
            do
                CUDA_VISIBLE_DEVICES='5' python main.py \
                    --data_dir $DATA_DIR \
                    --do_eval \
                    --output_dir $OUTPUT_DIR \
                    --model unified \
                    --metric $m \
                    --N $n \
                    --K $k \
                    --seed $i \
                    --train_epoch 10000 \
                    --eval_epoch 500 \
                    --eval_step 500 \
                    --test_epoch 3000
            done
        done
    done
done


for c in ${CRF[@]}
do
    for n in ${N[@]}
    do
        for k in ${K[@]}
        do
            CUDA_VISIBLE_DEVICES='5' python main.py \
                --data_dir $DATA_DIR \
                --do_train \
                --output_dir $OUTPUT_DIR \
                --max_len 128 \
                --learning_rate 5e-5 \
                --model $c \
                --N $n \
                --K $k \
                --train_epoch 10000 \
                --eval_epoch 500 \
                --eval_step 500 \
                --test_epoch 3000

            for i in {1..5}
            do
                CUDA_VISIBLE_DEVICES='5' python main.py \
                    --data_dir $DATA_DIR \
                    --do_eval \
                    --output_dir $OUTPUT_DIR \
                    --model $m \
                    --N $n \
                    --K $k \
                    --seed $i \
                    --train_epoch 10000 \
                    --eval_epoch 500 \
                    --eval_step 500 \
                    --test_epoch 3000
            done
        done
    done
done