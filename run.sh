source activate torch
export CUDA_VISIBLE_DEVICES=0
python run.py --data_dir /data/bzw/MRC/data/Squad_v1 \
    --model_name_or_path  /data/package/bert-base-uncased \
    --output_dir models \
    --model_type /data/package/bert-base-uncased \
    --train_file train.json \
    --predict_file dev.json \
    --overwrite_output_dir \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-5\
    --num_train_epochs 3 \
    --evaluate_during_training \
    --do_train \
    --output_result_dir ./models \
    --weight_decay 0.01
#    --overwrite_cache
#    --do_eval \
    
