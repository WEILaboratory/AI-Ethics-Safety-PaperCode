# With Loss-neutral Epsilon 0.1
for epoch in {0..49..1}
do
    model="main/debiased_models/bert-large-uncased/scm-neutral_epsilon_0_1/$epoch"
    echo "../$model/best_model_ckpt"
    CUDA_VISIBLE_DEVICES=5 nohup python -u stereoset.py \
        --model bert-large-uncased \
        --model_name_or_path ../$model/best_model_ckpt \
        --algorithm DPCE \
        --batch_size 1 \
        > ../log/evaluate/evaluate_scm-neutral_epsilon_0_1.log 2>&1 &
    wait
    echo ${model////_}
    wait
    nohup python -u stereoset_evaluation.py \
    --predictions_file "./results/stereoset/${model////_}.json" \
    > ../log/evaluate/res_scm-neutral_epsilon_0_1.log 2>&1 &
done
