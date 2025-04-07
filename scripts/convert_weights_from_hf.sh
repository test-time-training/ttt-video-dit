#!/bin/bash

FINAL_SAVE_PATH="TODO"
HUGGINGFACE_PRETRAINED_WEIGHTS_PATH="TODO"
SSM_TYPE="ttt_mlp" # Either ttt_linear or ttt_mlp

# Check if directory exists, prevent override
if [ -d "$FINAL_SAVE_PATH" ]; then
	echo "Warning: Overriding current weights at this path. If you wish to proceed, comment this check out."
	exit 1
else
	mkdir -p "$FINAL_SAVE_PATH"
fi

python -m ttt.models.cogvideo.weight_conversion.from_hf \
	--final_save_path ${FINAL_SAVE_PATH} \
	--ssm_type ${SSM_TYPE} \
	--pretrained_weights_dir ${HUGGINGFACE_PRETRAINED_WEIGHTS_PATH}

