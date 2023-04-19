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

FILE_TRANSFER_COMMAND=$FILE_TRANSFER_COMMAND
if [ -z "$FILE_TRANSFER_COMMAND" ]; then
  FILE_TRANSFER_COMMAND=scp # Only supports scp and lftp, use 'sudo apt-get install lftp' to install lftp, default is scp
fi

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# FT-2000+/64+XPU: TARGET_OS=linux and TARGET_ABI=arm64
# Intel-x86+XPU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
if [ -n "$5" ]; then
  TARGET_OS=$5
fi

WORK_SPACE="/var/tmp/test"
if [ "$TARGET_OS" == "android" ]; then
  WORK_SPACE=/data/local/tmp/test
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

SSH_DEVICE_IP_ADDR="192.168.180.8"
if [ -n "$8" ]; then
  SSH_DEVICE_IP_ADDR="$8"
fi

SSH_DEVICE_SSH_PORT="22"
if [ -n "$9" ]; then
  SSH_DEVICE_SSH_PORT="$9"
fi

SSH_DEVICE_USR_ID="toybrick"
if [ -n "${10}" ]; then
  SSH_DEVICE_USR_ID="${10}"
fi

SSH_DEVICE_USR_PWD="toybrick"
if [ -n "${11}" ]; then
  SSH_DEVICE_USR_PWD="${11}"
fi

#EXPORT_ENVIRONMENT_VARIABLES="export GLOG_v=5;"
EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH=./$DEVICE_NAME/paddle/lib:.:\$LD_LIBRARY_PATH;"
if [[ "$TARGET_ABI" = "amd64" ]]; then
  EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:./$DEVICE_NAME/third_party/install/mklml/lib:./$DEVICE_NAME/third_party/install/mkldnn/lib;"
fi
if [[ "$DEVICE_NAME" = "xpu" ]]; then
  EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export XPU_VISIBLE_DEVICES=0;export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:./$DEVICE_NAME/third_party/install/xpu/lib;"
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}.${DEVICE_NAME}

if [ "$FILE_TRANSFER_COMMAND" == "lftp" ]; then
  set -e
  lftp -e "rm -rf $WORK_SPACE; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "mkdir -p $WORK_SPACE; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; mirror -R ../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; mirror -R ../assets/models/$MODEL_NAME; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; put $BUILD_DIR/demo; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "cd $WORK_SPACE; ${EXPORT_ENVIRONMENT_VARIABLES} chmod +x ./demo; ./demo ./$MODEL_NAME $INPUT_SHAPES $INPUT_TYPES $OUTPUT_TYPES $DEVICE_NAME"
else
  set -e
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "rm -rf $WORK_SPACE"
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "mkdir -p $WORK_SPACE"
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../assets/models/$MODEL_NAME $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT $BUILD_DIR/demo $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "cd $WORK_SPACE; ${EXPORT_ENVIRONMENT_VARIABLES} chmod +x ./demo; ./demo ./$MODEL_NAME $INPUT_SHAPES $INPUT_TYPES $OUTPUT_TYPES $DEVICE_NAME"
fi
