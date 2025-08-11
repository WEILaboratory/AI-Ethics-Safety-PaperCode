CUDA_VISIBLE_DEVICES=0 nohup python -u ../debias_epsilon.py \
    --bias 'gender' \
    --weighted_loss 0.2 0.8 \
    --learning_rate 1e-4 \
    --lr_end 1e-7 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dev_data_size 3000 \
    --num_train_epochs 50 \
    --data_file '../sentences_collection_parallel/bert-large-uncased/gender_brief-scm/data.bin' \
    --output_dir '../debiased_models/bert-large-uncased/gender_brief-scm_epsilon_0_1' \
    --loss_neutral \
    > ../../log/debias/debias-gender_brief-scm_epsilon_0_1.log 2>&1 &