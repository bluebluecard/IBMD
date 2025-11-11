export PYTHONPATH=$PYTHONPATH:./

export CUDA_VISIBLE_DEVICES=0,1,2,3
run_args="--nproc_per_node 4 \
          --master_port 29513"

# Batch size per GPU
BS=256

DATASET_NAME=$1
FREQ_SAVE_ITER=100
if [[ $DATASET_NAME == "e2h" ]]; then
    MODEL_PATH=assets/ckpts/e2h_ema_0.9999_420000_adapted.pt
elif [[ $DATASET_NAME == "diode" ]]; then
    MODEL_PATH=assets/ckpts/diode_ema_0.9999_440000_adapted.pt
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    MODEL_PATH=assets/ckpts/imagenet256_inpaint_ema_0.9999_400000.pt
fi

source scripts/args.sh $DATASET_NAME

torchrun $run_args scripts/ddbm_train.py --exp=$EXP \
--schedule_sampler $SAMPLER --distillation=True --noise_schedule $PRED \
${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
--condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
--dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
--use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET \
${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} --resume_checkpoint=$MODEL_PATH \
--class_cond $CLASS_COND --use_scale_shift_norm True \
--ema_rate 0.99 --global_batch_size $BS \
--lr 0.000003 --num_head_channels 64 \
--resblock_updown True --microbatch=$MICRO_BS \
--use_fp16 $USE_16FP --weight_decay 0.0 \
--sigma_data $SIGMA_DATA --cov_xy $COV_XY \
--num_workers 4 \
--save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False --n_bridge_loop 5 \
--resume_student_checkpoint "" \
--add_noise True --noise_channels 1 --num_steps 2