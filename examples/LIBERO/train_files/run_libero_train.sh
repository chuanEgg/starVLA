

# For single GPU training, disable NCCL network configuration
# Uncomment below lines for multi-GPU/multi-node training
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
run_id=0311_libero_test_qwenGR00T_0.8b
run_root_dir=/tmp2/chuan/starVLA_results
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 10 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
