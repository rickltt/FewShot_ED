CUDA_VISIBLE_DEVICES=2 python run.py \
    --data_dir ./data/fewevent \
    --model unified \
    --metric dot \
    --output few_output \
    --trainN 5 \
    --evalN 5 \
    --K 5 \
    --do_train \
    --do_eval 