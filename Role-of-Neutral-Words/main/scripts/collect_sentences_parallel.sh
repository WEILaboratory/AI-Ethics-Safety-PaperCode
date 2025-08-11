model_type=$1
data=$2
bias=$3
block_size=128

if [ $model_type == 'bert' ]; then
    model_name_or_path=bert-large-uncased
elif [ $model_type == 'roberta' ]; then
    model_name_or_path=roberta-large
fi

neutral_words='../../data/neutral.txt'

if [ $bias == 'gender' ]; then
    attribute_words='../../data/female.txt,../../data/male.txt'
elif [ $bias == 'gender_brief-scm' ]; then
    attribute_words='../../data/female_brief.txt,../../data/male_brief.txt'
    neutral_words='../../data/warm.txt,../../data/cold.txt,../../data/competent.txt,../../data/incompetent.txt'
elif [ $bias == 'gender_brief' ]; then
    attribute_words='../../data/female_brief.txt,../../data/male_brief.txt'
elif [ $bias == 'religion' ]; then
    attribute_words='../../data/judaism.txt,../../data/christianity.txt,../../data/islam.txt'
elif [ $bias == 'scm' ]; then
    attribute_words='../../data/warm.txt,../../data/cold.txt,../../data/competent.txt,../../data/incompetent.txt'
elif [ $bias == 'warmth' ]; then
    attribute_words='../../data/warm.txt,../../data/cold.txt'
elif [ $bias == 'competence' ]; then
    attribute_words='../../data/competent.txt,../../data/incompetent.txt'
fi

OUTPUT_DIR=../sentences_collection_parallel/$model_name_or_path/$bias

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
echo $model_type $bias

CUDA_VISIBLE_DEVICES=7 \
python -u ../collect_sentences_parallel.py \
    --input ../../data/$data \
    --neutral_words $neutral_words \
    --attribute_words $attribute_words \
    --output $OUTPUT_DIR \
    --block_size $block_size \
    --model_type $model_type \
    --ab_test_type 'final' > ../../log/collect-sentence/collect-$bias.log
