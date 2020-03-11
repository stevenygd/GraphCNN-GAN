#!/bin/bash

models=("gconv_up_aggr")
pc_classes=("airplane" "chair" "sofa" "table")

this_folder=`pwd`

for class_name in ${pc_classes[@]}; do
	for model in ${models[@]}; do
        echo $class_name;
		render_dir="$this_folder/Results/$model/$class_name/renders/";
		save_dir="$this_folder/Results/$model/$class_name/saved_models/";
        mkdir -p $render_dir;
		CUDA_VISIBLE_DEVICES=0 python "$model""_code/test.py" --class_name $class_name --render_dir $render_dir --save_dir $save_dir;

	done;
done
