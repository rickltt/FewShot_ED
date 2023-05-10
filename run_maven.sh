CUDA_VISIBLE_DEVICES=3 python run.py \
    --data_dir ./data/maven \
    --model unified \
    --metric dot \
    --output maven_output \
    --trainN 5 \
    --evalN 5 \
    --K 5 \
    --do_train \
    --do_eval 