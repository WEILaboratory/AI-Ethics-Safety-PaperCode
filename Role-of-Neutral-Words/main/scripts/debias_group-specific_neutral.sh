# gender-specific neutral words
# CUDA_VISIBLE_DEVICES=7 nohup python -u ../debias_epsilon_group-specific_neutral.py \
#     --bias 'scm' \
#     --weighted_loss 0.2 0.8 \
#     --learning_rate 1e-4 \
#     --lr_end 1e-7 \
#     --per_device_train_batch_size 128 \
#     --per_device_eval_batch_size 128 \
#     --dev_data_size 500 \
#     --num_train_epochs 30 \
#     --data_file '../sentences_collection_parallel_group-specific_neutral/bert-large-uncased/scm/data.bin' \
#     --output_dir '../debiased_models/bert-large-uncased/scm-gender-specific-neutral_epsilon_0_3' \
#     --loss_neutral \
#     > ../../log/debias/debias-scm-gender-specific-neutral_epsilon_0_3.log 2>&1 &

# # religion-specific neutral words
CUDA_VISIBLE_DEVICES=7 nohup python -u ../debias_epsilon_group-specific_neutral.py \
    --bias 'scm' \
    --weighted_loss 0.2 0.8 \
    --learning_rate 1e-4 \
    --lr_end 1e-7 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dev_data_size 500 \
    --num_train_epochs 30 \
    --data_file '../sentences_collection_parallel_group-specific_neutral/bert-large-uncased/scm/data.bin' \
    --output_dir '../debiased_models/bert-large-uncased/scm-religion-specific-neutral_epsilon_0_3' \
    --loss_neutral \
    > ../../log/debias/debias-scm-religion-specific-neutral_epsilon_0_3.log 2>&1 &