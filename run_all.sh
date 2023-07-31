#!/bin/bash

OUTPUT_DIR='./output'
DATASET=(./data/ere ./data/ace ./data/fewevent ./data/maven)
N=(5 10)
K=(5 10)
for d in ${DATASET[@]}
do
    for n in ${N[@]}
    do
        for k in ${K[@]}
        do
            CUDA_VISIBLE_DEVICES='6' python main.py \
                --data_dir $d \
                --do_train \
                --output_dir $OUTPUT_DIR \
                --learning_rate 3e-5 \
                --N $n \
                --K $k \
                --train_epoch 10000 \
                --eval_epoch 500 \
                --eval_step 500 \
                --test_epoch 3000
            for i in {1..5}
            do
                CUDA_VISIBLE_DEVICES='6' python main.py \
                    --data_dir $d \
                    --do_eval \
                    --output_dir $OUTPUT_DIR \
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
