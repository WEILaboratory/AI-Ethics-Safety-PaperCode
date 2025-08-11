# religion-specific neutral words
for epoch in {0..49..1}
do
    model="main/debiased_models/bert-large-uncased/scm-religion-specific-neutral_epsilon_0_3/$epoch"
    echo "../$model/best_model_ckpt"
    CUDA_VISIBLE_DEVICES=3 nohup python -u stereoset.py \
        --model bert-large-uncased \
        --model_name_or_path ../$model/best_model_ckpt \
        --algorithm DPCE \
        --batch_size 1 \
        > ../log/evaluate/scm-religion-specific-neutral_epsilon_0_3.log 2>&1 &
    wait
    echo ${model////_}
    wait
    nohup python -u stereoset_evaluation.py \
    --predictions_file "./results/stereoset/${model////_}.json" \
    > ../log/evaluate/res_scm-religion-specific-neutral_epsilon_0_3.log 2>&1 &
done

# # gender-specific neutral words
# for epoch in {0..49..1}
# do
#     model="main/debiased_models/bert-large-uncased/scm-gender-specific-neutral_epsilon_0_3/$epoch"
#     echo "../$model/best_model_ckpt"
#     CUDA_VISIBLE_DEVICES=4 nohup python -u stereoset.py \
#         --model bert-large-uncased \
#         --model_name_or_path ../$model/best_model_ckpt \
#         --algorithm DPCE \
#         --batch_size 1 \
#         > ../log/evaluate/scm-gender-specific-neutral_epsilon_0_3.log 2>&1 &
#     wait
#     echo ${model////_}
#     wait
#     nohup python -u stereoset_evaluation.py \
#     --predictions_file "./results/stereoset/${model////_}.json" \
#     > ../log/evaluate/res_scm-gender-specific-neutral_epsilon_0_3.log 2>&1 &
# done
