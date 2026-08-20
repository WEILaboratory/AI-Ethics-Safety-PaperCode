rootdir=[root_to_watertight_models]
categories="$(ls $rootdir)"
for category in ${categories[@]}
do
    echo "$rootdir/$category/2_watertight/"
    file_list=$(ls $rootdir/$category/2_watertight/*.off)
    if [ $? -ne 0 ]; then
        continue
    fi
    for file in ${file_list[@]}
    do
        echo $file
        python prepare_batch.py --model $file --output_dir data
    done
done
