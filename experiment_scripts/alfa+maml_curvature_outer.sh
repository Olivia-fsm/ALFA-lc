#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

# put the path to the main directory here
# cd ..
cd /mlodata1/sfan/optML_proj/ALFA

export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$1,$2,$3,$4
# Activate the relevant virtual environment:
python train_maml_system.py --name_of_args_json_file experiment_config/alfa+maml_curvature_outer.json --gpu_to_use $GPU_ID
