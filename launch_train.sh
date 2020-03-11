#!/bin/bash

models=("gconv_up_aggr")
# pc_classes=("chair")
pc_classes=("airplane")

this_folder=`pwd`
export start_iter=1
for class_name in ${pc_classes[@]}; do
	for model in ${models[@]}; do

		render_dir="$this_folder/Results/${model}_v2/$class_name/renders/"
		log_dir="$this_folder/log_dir/${model}_v2/$class_name/"
		save_dir="$this_folder/Results/${model}_v2/$class_name/saved_models/"
        mkdir -p $render_dir $save_dir $log_dir
		CUDA_VISIBLE_DEVICES=1 python "$model""_code/main.py" --class_name $class_name --start_iter $start_iter --render_dir $render_dir --log_dir $log_dir --save_dir $save_dir

	done
done
