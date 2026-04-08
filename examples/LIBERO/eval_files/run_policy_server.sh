#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/chuanchen/miniconda3/envs/starVLA/bin/python
your_ckpt=/tmp2/chuan/starVLA_results/0311_libero_test_qwenGR00T_0.8b/final_model/pytorch_model.pt
gpu_id=1
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
