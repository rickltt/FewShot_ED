CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_dir ./data/ace \
    --model unified \
    --metric dot \
    --output ace_output \
    --trainN 5 \
    --evalN 5 \
    --K 5 \
    --do_train \
    --do_eval 