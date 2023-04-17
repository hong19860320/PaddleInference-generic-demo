#!/bin/bash
MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024
#MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer
#MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel
if [ -n "$1" ]; then
  MODEL_NAME=$1
fi
if [ ! -d "../assets/models/$MODEL_NAME" ];then
  MODEL_URL="http://paddlelite-demo.bj.bcebos.com/devices/generic/models/${MODEL_NAME}.tar.gz"
  echo "Model $MODEL_NAME not found! Try to download it from $MODEL_URL ..."
  curl $MODEL_URL -o -| tar -xz -C ../assets/models
  if [[ $? -ne 0 ]]; then
    echo "Model $MODEL_NAME download failed!"
    exit 1
  fi
fi

CONFIG_NAME=cityscapes_512_1024.txt
#CONFIG_NAME=human_224_398.txt
if [ -n "$2" ]; then
  CONFIG_NAME=$2
fi

DATASET_NAME=test
if [ -n "$3" ]; then
  DATASET_NAME=$3
fi

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# FT-2000+/64+XPU: TARGET_OS=linux and TARGET_ABI=arm64
# Intel-x86+XPU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
if [ -n "$4" ]; then
  TARGET_OS=$4
fi

TARGET_ABI=arm64
if [ -n "$5" ]; then
  TARGET_ABI=$5
fi

# FT-2000+/64+XPU, Intel-x86+XPU: DEVICE_NAME=xpu
# CPU only: DEVICE_NAME=cpu
DEVICE_NAME="cpu"
if [ -n "$6" ]; then
  DEVICE_NAME="$6"
fi

#export GLOG_v=5
export LD_LIBRARY_PATH=../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/cpu/paddle/lib:../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME/paddle/lib:.:$LD_LIBRARY_PATH
if [[ "$DEVICE_NAME" = "xpu" ]]; then
  export XPU_VISIBLE_DEVICES=0
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}.${DEVICE_NAME}

set -e
chmod +x ./$BUILD_DIR/demo
./$BUILD_DIR/demo ../assets/models/$MODEL_NAME ../assets/configs/$CONFIG_NAME ../assets/datasets/$DATASET_NAME $DEVICE_NAME
