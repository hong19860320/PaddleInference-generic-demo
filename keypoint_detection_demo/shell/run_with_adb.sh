#!/bin/bash
MODEL_NAME=tinypose_fp32_128_96
#MODEL_NAME=tinypose_fp32_256_192
#MODEL_NAME=tinypose_int8_128_96_per_channel
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

CONFIG_NAME=tinypose_128_96.txt
#CONFIG_NAME=tinypose_256_192.txt
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
TARGET_OS=android
if [ -n "$4" ]; then
  TARGET_OS=$4
fi

WORK_SPACE=/data/local/tmp/test
if [ "$TARGET_OS" == "linux" ]; then
  WORK_SPACE=/var/tmp/test
fi

TARGET_ABI=arm64-v8a
if [ -n "$5" ]; then
  TARGET_ABI=$5
fi

# FT-2000+/64+XPU, Intel-x86+XPU: DEVICE_NAME=xpu
# CPU only: DEVICE_NAME=cpu
DEVICE_NAME="cpu"
if [ -n "$6" ]; then
  DEVICE_NAME="$6"
fi

ADB_DEVICE_NAME=
if [ -n "$7" ]; then
  ADB_DEVICE_NAME="-s $7"
fi

EXPORT_ENVIRONMENT_VARIABLES="export GLOG_v=5;"
EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH=./$DEVICE_NAME/paddle/lib:.:\$LD_LIBRARY_PATH;"

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}.${DEVICE_NAME}

set -e
adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE"
adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE"
adb $ADB_DEVICE_NAME push ../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME $WORK_SPACE
adb $ADB_DEVICE_NAME push ../assets/models/$MODEL_NAME $WORK_SPACE
adb $ADB_DEVICE_NAME push ../assets/configs/. $WORK_SPACE
adb $ADB_DEVICE_NAME push $BUILD_DIR/demo $WORK_SPACE
COMMAND_LINE="cd $WORK_SPACE; $EXPORT_ENVIRONMENT_VARIABLES chmod +x ./demo; ./demo ./$MODEL_NAME ./$CONFIG_NAME ./$DATASET_NAME $DEVICE_NAME"
rm -rf ../assets/datasets/$DATASET_NAME/outputs
mkdir -p ../assets/datasets/$DATASET_NAME/outputs
SPLIT_COUNT=200
SPLIT_INDEX=0
SAMPLE_INDEX=0
SAMPLE_START=0
for SAMPLE_NAME in $(cat ../assets/datasets/$DATASET_NAME/list.txt); do
  echo $SAMPLE_INDEX + ": " + $SAMPLE_NAME
  if [ $SAMPLE_INDEX -ge $SAMPLE_START ] ; then 
    if [ $SPLIT_INDEX -eq $SPLIT_COUNT ] ; then
      adb $ADB_DEVICE_NAME push list.txt $WORK_SPACE/$DATASET_NAME/
      adb $ADB_DEVICE_NAME shell "$COMMAND_LINE"
      adb $ADB_DEVICE_NAME pull $WORK_SPACE/$DATASET_NAME/outputs/ ../assets/datasets/$DATASET_NAME/outputs/
      SPLIT_INDEX=0
    fi
    if [ $SPLIT_INDEX -eq 0 ] ; then 
      adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE/$DATASET_NAME/inputs"
      adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$DATASET_NAME/inputs"
      adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE/$DATASET_NAME/outputs"
      adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$DATASET_NAME/outputs"
      rm -rf list.txt
    fi
    adb $ADB_DEVICE_NAME push ../assets/datasets/$DATASET_NAME/inputs/$SAMPLE_NAME $WORK_SPACE/$DATASET_NAME/inputs/
    echo -e "$SAMPLE_NAME" >> list.txt
    SPLIT_INDEX=$(($SPLIT_INDEX + 1))
  else
    echo "skip..."
  fi 
  SAMPLE_INDEX=$(($SAMPLE_INDEX + 1))
done
if [ $SPLIT_INDEX -gt 0 ] ; then
  adb $ADB_DEVICE_NAME push list.txt $WORK_SPACE/$DATASET_NAME/
  adb $ADB_DEVICE_NAME shell "$COMMAND_LINE"
  adb $ADB_DEVICE_NAME pull $WORK_SPACE/$DATASET_NAME/outputs/ ../assets/datasets/$DATASET_NAME/outputs/
fi
rm -rf list.txt
