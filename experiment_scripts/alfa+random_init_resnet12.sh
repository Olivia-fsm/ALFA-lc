#!/bin/sh

export GPU_ID=0

echo $GPU_ID

cd /mlodata1/sfan/optML_proj/ALFA-plus
pip install -r requirements.txt
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Activate the relevant virtual environment:
python train_maml_system.py --name_of_args_json_file experiment_config/alfa+random_init_resnet12.json --gpu_to_use $GPU_ID
python train_maml_system.py --name_of_args_json_file experiment_config/alfa+random_init_resnet12_outer.json --gpu_to_use $GPU_ID
python train_maml_system.py --name_of_args_json_file experiment_config/alfa+random_init_resnet12_clipping.json --gpu_to_use $GPU_ID
