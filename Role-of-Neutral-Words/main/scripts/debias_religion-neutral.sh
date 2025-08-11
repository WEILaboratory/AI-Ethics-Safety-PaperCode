# No Loss-neutral - religon
CUDA_VISIBLE_DEVICES=5 nohup python -u ../debias.py \
    --bias 'religion' \
    --weighted_loss 0.2 0.8 \
    --learning_rate 1e-4 \
    --lr_end 1e-7 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dev_data_size 300 \
    --num_train_epochs 50 \
    --data_file '../sentences_collection_parallel/bert-large-uncased/religion/data.bin' \
    --output_dir '../debiased_models/bert-large-uncased/religion-neutral-no-loss_neutral' \
    > ../../log/debias/debias-religion-neutral-no-loss_neutral.log 2>&1 &

# With Loss-neutral - gender_brief
# CUDA_VISIBLE_DEVICES=6 nohup python -u ../debias.py \
#     --bias 'gender' \
#     --weighted_loss 0.2 0.8 \
#     --learning_rate 1e-4 \
#     --lr_end 1e-7 \
#     --per_device_train_batch_size 128 \
#     --per_device_eval_batch_size 128 \
#     --dev_data_size 3000 \
#     --num_train_epochs 50 \
#     --data_file '../sentences_collection_parallel/bert-large-uncased/gender_brief/data.bin' \
#     --output_dir '../debiased_models/bert-large-uncased/gender_brief-neutral' \
#     --loss_neutral \
#     > ../../log/debias/debias-gender_brief-neutral.log 2>&1 &
