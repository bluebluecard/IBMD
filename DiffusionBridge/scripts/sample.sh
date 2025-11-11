export PYTHONPATH=$PYTHONPATH:./

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
export CUDA_VISIBLE_DEVICES=0
run_args="--nproc_per_node 1 \
          --master_port 29512"


# Dataset and checkpoint
DATASET_NAME=$1

if [[ $DATASET_NAME == "e2h" ]]; then
    SPLIT=test
    BS=20
    MODEL_PATH=PATH_TO_YOUR_MODEL
elif [[ $DATASET_NAME == "diode" ]]; then
    SPLIT=test
    BS=20
    MODEL_PATH=PATH_TO_YOUR_MODEL
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    BS=64
    SPLIT=test
    MODEL_PATH=PATH_TO_YOUR_MODEL
fi

source scripts/args.sh $DATASET_NAME

# Number of function evaluations (NFE)
NFE=$2

torchrun $run_args sample.py --sampler $GEN_SAMPLER --batch_size $BS \
--class_cond $CLASS_COND --noise_schedule $PRED \
${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
--condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
--dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
--use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT\
${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \
${ETA:+ --eta="${ETA}"} \
${ORDER:+ --order="${ORDER}"} --add_noise True --noise_channels 1 --num_steps $NFE \
--model_path $MODEL_PATH