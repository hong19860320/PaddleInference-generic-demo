#!/bin/bash
#MODEL_NAME=conv_add_relu_dwconv_add_relu_224_int8_per_layer
#MODEL_NAME=conv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x4_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_mul_add_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_conv_add_relu_dwconv_add_relu_224_int8_per_channel
MODEL_NAME=conv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_mul_add_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_res2c_224_fp32
INPUT_SHAPES="1,3,224,224"
INPUT_TYPES="float32"
OUTPUT_TYPES="float32"

#MODEL_NAME=conv_add_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_mul_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_mul_144_192_int8_per_layer
#INPUT_SHAPES="1,3,192,144"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

#MODEL_NAME=eltwise_mul_broadcast_per_layer
#INPUT_SHAPES="1,3,384,384"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

#MODEL_NAME=dwconv_ic_128_groups_128_oc_256_per_layer
#INPUT_SHAPES="1,3,320,320"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

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

if [ -n "$2" ]; then
  INPUT_SHAPES=$2
fi

if [ -n "$3" ]; then
  INPUT_TYPES=$3
fi

if [ -n "$4" ]; then
  OUTPUT_TYPES=$4
fi

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# FT-2000+/64+XPU: TARGET_OS=linux and TARGET_ABI=arm64
# Intel-x86+XPU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
if [ -n "$5" ]; then
  TARGET_OS=$5
fi

TARGET_ABI=arm64
if [ -n "$6" ]; then
  TARGET_ABI=$6
fi

# FT-2000+/64+XPU, Intel-x86+XPU: DEVICE_NAME=xpu
# CPU only: DEVICE_NAME=cpu
DEVICE_NAME="cpu"
if [ -n "$7" ]; then
  DEVICE_NAME="$7"
fi

#export GLOG_v=5
export LD_LIBRARY_PATH=../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/cpu/paddle/lib:../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME/paddle/lib:.:$LD_LIBRARY_PATH
if [[ "$DEVICE_NAME" = "xpu" ]]; then
  export XPU_VISIBLE_DEVICES=0
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}.${DEVICE_NAME}

set -e
chmod +x ./$BUILD_DIR/demo
./$BUILD_DIR/demo ../assets/models/$MODEL_NAME $INPUT_SHAPES $INPUT_TYPES $OUTPUT_TYPES $DEVICE_NAME
