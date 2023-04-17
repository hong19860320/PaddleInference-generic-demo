#!/bin/bash
set -e

# Settings only for Android
ANDROID_NDK=/opt/android-ndk-r17c # docker
#ANDROID_NDK=/Users/hongming/Library/android-ndk-r17c # macOS

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# FT-2000+/64+XPU: TARGET_OS=linux and TARGET_ABI=arm64
# Intel-x86+XPU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
#TARGET_OS=android
#TARGET_OS=qnx # cmake 3.22.3 or later required
if [ -n "$1" ]; then
  TARGET_OS=$1
fi

TARGET_ABI=arm64
if [ -n "$2" ]; then
  TARGET_ABI=$2
fi

# FT-2000+/64+XPU, Intel-x86+XPU: DEVICE_NAME=xpu
# CPU only: DEVICE_NAME=cpu
DEVICE_NAME=cpu
if [ -n "$3" ]; then
  DEVICE_NAME=$3
fi

function readlinkf() {
  perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

CMAKE_COMMAND_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_SKIP_RPATH=ON -DTARGET_OS=${TARGET_OS} -DTARGET_ABI=${TARGET_ABI} -DDEVICE_NAME=${DEVICE_NAME} -DPADDLE_INFERENCE_DIR=$(readlinkf ../../libs/PaddleInference) -DOpenCV_DIR=$(readlinkf ../../libs/OpenCV)"
if [ "${TARGET_OS}" == "android" ]; then
  ANDROID_NATIVE_API_LEVEL=android-23
  if [ $TARGET_ABI == "armeabi-v7a" ]; then
    ANDROID_NATIVE_API_LEVEL=android-21
  fi
  CMAKE_COMMAND_ARGS="${CMAKE_COMMAND_ARGS} -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_NDK=${ANDROID_NDK} -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} -DANDROID_STL=c++_shared -DANDROID_ABI=${TARGET_ABI} -DANDROID_ARM_NEON=TRUE"
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}.${DEVICE_NAME}

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake ${CMAKE_COMMAND_ARGS} ..
make
