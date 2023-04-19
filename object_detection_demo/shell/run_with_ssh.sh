#!/bin/bash
MODEL_NAME=ssd_mobilenet_v1_relu_voc_fp32_300
#MODEL_NAME=ssd_mobilenet_v1_relu_voc_int8_300_per_layer
#MODEL_NAME=yolov3_mobilenet_v1_270e_coco_fp32_608
#MODEL_NAME=yolov3_darknet53_270e_coco_fp32_608
#MODEL_NAME=picodet_relu6_int8_416_per_channel
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

CONFIG_NAME=ssd_voc_300.txt
#CONFIG_NAME=yolov3_coco_608.txt
#CONFIG_NAME=picodet_coco_416.txt
if [ -n "$2" ]; then
  CONFIG_NAME=$2
fi

DATASET_NAME=test
if [ -n "$3" ]; then
  DATASET_NAME=$3
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
if [ -n "$4" ]; then
  TARGET_OS=$4
fi

WORK_SPACE="/var/tmp/test"
if [ "$TARGET_OS" == "android" ]; then
  WORK_SPACE=/data/local/tmp/test
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

SSH_DEVICE_IP_ADDR="192.168.180.8"
if [ -n "$7" ]; then
  SSH_DEVICE_IP_ADDR="$7"
fi

SSH_DEVICE_SSH_PORT="22"
if [ -n "$8" ]; then
  SSH_DEVICE_SSH_PORT="$8"
fi

SSH_DEVICE_USR_ID="toybrick"
if [ -n "$9" ]; then
  SSH_DEVICE_USR_ID="$9"
fi

SSH_DEVICE_USR_PWD="toybrick"
if [ -n "${10}" ]; then
  SSH_DEVICE_USR_PWD="${10}"
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

COMMAND_LINE="cd $WORK_SPACE; $EXPORT_ENVIRONMENT_VARIABLES chmod +x ./demo; ./demo ./$MODEL_NAME ./$CONFIG_NAME ./$DATASET_NAME $DEVICE_NAME"
rm -rf ../assets/datasets/$DATASET_NAME/outputs
mkdir -p ../assets/datasets/$DATASET_NAME/outputs
SPLIT_COUNT=200
SPLIT_INDEX=0
SAMPLE_INDEX=0
SAMPLE_START=0
if [ "$FILE_TRANSFER_COMMAND" == "lftp" ]; then
  set -e
  lftp -e "rm -rf $WORK_SPACE; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "mkdir -p $WORK_SPACE; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; mirror -R ../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; mirror -R ../assets/models/$MODEL_NAME; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  lftp -e "cd $WORK_SPACE; mput ../assets/configs/*; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR:
  lftp -e "cd $WORK_SPACE; put $BUILD_DIR/demo; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  for SAMPLE_NAME in $(cat ../assets/datasets/$DATASET_NAME/list.txt); do
    echo $SAMPLE_INDEX + ": " + $SAMPLE_NAME
    if [ $SAMPLE_INDEX -ge $SAMPLE_START ] ; then
      if [ $SPLIT_INDEX -eq $SPLIT_COUNT ] ; then
        lftp -e "cd $WORK_SPACE/$DATASET_NAME/; put list.txt; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "${COMMAND_LINE}"
        lftp -e "set xfer:clobber on; cd $WORK_SPACE; mirror $DATASET_NAME/outputs ../assets/datasets/$DATASET_NAME/outputs/; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        SPLIT_INDEX=0
      fi
      if [ $SPLIT_INDEX -eq 0 ] ; then
        lftp -e "rm -rf $WORK_SPACE/$DATASET_NAME/inputs; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        lftp -e "mkdir -p $WORK_SPACE/$DATASET_NAME/inputs; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        lftp -e "rm -rf $WORK_SPACE/$DATASET_NAME/outputs; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        lftp -e "mkdir -p $WORK_SPACE/$DATASET_NAME/outputs; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
        rm -rf list.txt
      fi
      lftp -e "cd $WORK_SPACE/$DATASET_NAME/inputs/; put ../assets/datasets/$DATASET_NAME/inputs/$SAMPLE_NAME; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
      echo -e "$SAMPLE_NAME" >> list.txt
      SPLIT_INDEX=$(($SPLIT_INDEX + 1))
    else
      echo "skip..."
    fi 
    SAMPLE_INDEX=$(($SAMPLE_INDEX + 1))
  done
  if [ $SPLIT_INDEX -gt 0 ] ; then
    lftp -e "cd $WORK_SPACE/$DATASET_NAME/; put list.txt; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
    sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "${COMMAND_LINE}"
    lftp -e "set xfer:clobber on; cd $WORK_SPACE; mirror $DATASET_NAME/outputs ../assets/datasets/$DATASET_NAME/outputs/; bye" -u $SSH_DEVICE_USR_ID,$SSH_DEVICE_USR_PWD $SSH_DEVICE_IP_ADDR
  fi
  rm -rf list.txt
else
  set -e
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "rm -rf $WORK_SPACE"
  sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "mkdir -p $WORK_SPACE"
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../../libs/PaddleInference/$TARGET_OS/$TARGET_ABI/$DEVICE_NAME $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../assets/models/${MODEL_NAME} $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../assets/configs/* $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  sshpass -p $SSH_DEVICE_USR_PWD scp -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT $BUILD_DIR/demo $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE
  for SAMPLE_NAME in $(cat ../assets/datasets/$DATASET_NAME/list.txt); do
    echo $SAMPLE_INDEX + ": " + $SAMPLE_NAME
    if [ $SAMPLE_INDEX -ge $SAMPLE_START ] ; then
      if [ $SPLIT_INDEX -eq $SPLIT_COUNT ] ; then
        sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT list.txt $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE/$DATASET_NAME/
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "${COMMAND_LINE}"
        sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE/$DATASET_NAME/outputs/* ../assets/datasets/$DATASET_NAME/outputs/
        SPLIT_INDEX=0
      fi
      if [ $SPLIT_INDEX -eq 0 ] ; then
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "rm -rf $WORK_SPACE/$DATASET_NAME/inputs"
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "mkdir -p $WORK_SPACE/$DATASET_NAME/inputs"
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "rm -rf $WORK_SPACE/$DATASET_NAME/outputs"
        sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "mkdir -p $WORK_SPACE/$DATASET_NAME/outputs"
        rm -rf list.txt
      fi
      sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT ../assets/datasets/$DATASET_NAME/inputs/$SAMPLE_NAME $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE/$DATASET_NAME/inputs/
      echo -e "$SAMPLE_NAME" >> list.txt
      SPLIT_INDEX=$(($SPLIT_INDEX + 1))
    else
      echo "skip..."
    fi 
    SAMPLE_INDEX=$(($SAMPLE_INDEX + 1))
  done
  if [ $SPLIT_INDEX -gt 0 ] ; then
    sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT list.txt $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE/$DATASET_NAME/
    sshpass -p $SSH_DEVICE_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR "${COMMAND_LINE}"
    sshpass -p $SSH_DEVICE_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_DEVICE_SSH_PORT $SSH_DEVICE_USR_ID@$SSH_DEVICE_IP_ADDR:$WORK_SPACE/$DATASET_NAME/outputs/* ../assets/datasets/$DATASET_NAME/outputs/
  fi
  rm -rf list.txt
fi
