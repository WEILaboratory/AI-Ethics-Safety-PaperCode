# No Loss-neutral religion
for epoch in {0..49..1}
do
    model="main/debiased_models/bert-large-uncased/religion-neutral-no-loss_neutral/$epoch"
    echo "../$model/best_model_ckpt"
    CUDA_VISIBLE_DEVICES=7 nohup python -u stereoset.py \
        --model bert-large-uncased \
        --model_name_or_path ../$model/best_model_ckpt \
        --algorithm DPCE \
        --batch_size 1 \
        > ../log/evaluate/evaluate_religion-neutral-no-loss_neutral.log 2>&1 &
    wait
    echo ${model////_}
    wait
    nohup python -u stereoset_evaluation.py \
    --predictions_file "./results/stereoset/${model////_}.json" \
    > ../log/evaluate/res_religion-neutral-no-loss_neutral.log 2>&1 &
done
# With Loss-neutral
# for epoch in {0..49..1}
# do
#     model="main/debiased_models/bert-large-uncased/gender_brief-neutral/$epoch"
#     echo "../$model/best_model_ckpt"
#     CUDA_VISIBLE_DEVICES=6 nohup python -u stereoset.py \
#         --model bert-large-uncased \
#         --model_name_or_path ../$model/best_model_ckpt \
#         --algorithm DPCE \
#         --batch_size 1 \
#         > ../log/evaluate/evaluate_gender_brief-neutral.log 2>&1 &
#     wait
#     echo ${model////_}
#     wait
#     nohup python -u stereoset_evaluation.py \
#     --predictions_file "./results/stereoset/${model////_}.json" \
#     > ../log/evaluate/res_gender_brief-neutral.log 2>&1 &
# done
