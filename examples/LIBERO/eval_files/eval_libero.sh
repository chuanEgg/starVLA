#!/bin/bash

cd /home/chuanchen/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/tmp2/chuan/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/chuanchen/miniconda3/envs/starVLA_libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/tmp2/chuan/starVLA_results/0311_libero_test_qwenGR00T_0.8b/final_model/pytorch_model.pt
export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="/tmp2/chuan/starVLA_eval/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="/tmp2/chuan/starVLA_eval/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
