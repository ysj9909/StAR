#!/bin/bash

BASE_MODEL_PATH="pretrained_models/Qwen3-VL-8B-Instruct"
REASONING_MODEL_PATH="pretrained_models/StAR-8B/huggingface"

MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*pretrained_models\/(.*)\/actor\/.*/\1/')
# TEST_DATA_PATH="sj9909/MMR_val"
# TEST_DATA_PATH="sj9909/MMR_test_obj"
# TEST_DATA_PATH="sj9909/MMR_test_part"
# TEST_DATA_PATH="sj9909/MMR_test_mixed"

TEST_DATA_PATH="sj9909/MUSE_val"
# TEST_DATA_PATH="sj9909/MUSE_test_few"
# TEST_DATA_PATH="sj9909/MUSE_test_many"

TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
# OUTPUT_PATH="./outputs/mmr_eval_results/${MODEL_DIR}/${TEST_NAME}"
OUTPUT_PATH="./outputs/muse_eval_results/${MODEL_DIR}/${TEST_NAME}"

NUM_PARTS=8
# Create output directory
mkdir -p $OUTPUT_PATH

echo "REASONING_MODEL_PATH: $REASONING_MODEL_PATH"
echo "OUTPUT_PATH: '$OUTPUT_PATH'"

# Run 8 processes in parallel
for idx in {0..7}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation_star.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --vl_model_version qwen3 \
        --qwen3_base_path $BASE_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --use_lora true \
        --visualization false \
        --use_majority_voting false \
        --num_samples 32 \
        --test_muse_mmr true \
        --batch_size 32 &
done

# Wait for all processes to complete
wait

echo "MODEL_DIR: '$MODEL_DIR'"
echo "Test data path: '$TEST_DATA_PATH'"

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH