#!/bin/bash
set -e

source settings.sh

build_and_update_lib() {
  local os=$1
  local arch=$2
  local rebuild=$3
  local device_name=$4
  if [ "$device_name" = "" ]; then
    echo "device_name is empty!"
    return -1
  fi
  build_cmd="cmake .. -DPY_VERSION=3.7 -DPYTHON_EXECUTABLE=`which python3` -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF"
  extra_args=""
  if [ "$os" = "android" ]; then
    # Android
    if [ "$arch" = "armv8" ]; then
      lib_abi="arm64-v8a"
    elif [ "$arch" = "armv7" ]; then
      lib_abi="armeabi-v7a"
    else
      echo "Abi $arch is not supported for $os and any devices."
      return -1
    fi
    lib_os="android"
  elif [ "$os" = "linux" ]; then
    # linux
    if [ "$arch" = "armv8" ]; then
      lib_abi="arm64"
      if [ "$device_name" == "xpu" ]; then
        extra_args="-DWITH_XPU=ON"
      fi
    elif [ "$arch" = "armv7hf" ]; then
      lib_abi="armhf"
    elif [ "$arch" = "x86" ]; then
      lib_abi="amd64"
      extra_args="-DWITH_MKL=ON"
      if [ "$device_name" == "xpu" ]; then
        extra_args="-DWITH_XPU=ON"
      fi
    else
      echo "Abi $arch is not supported for $os and any devices."
      return -1
    fi
    lib_os="linux"
  else
    # qnx
    if [ "$arch" = "armv8" ]; then
      lib_abi="arm64"
    else
      echo "Abi $arch is not supported for $os and any devices."
      return -1
    fi
    lib_os="qnx"
  fi
  if [ "$arch" = "armv8" ] || [ "$arch" = "armv7hf" ]; then
    extra_args="$extra_args -DWITH_ARM=ON -DWITH_ARM_DNN_LIBRARY=ON -DARM_DNN_LIBRARY_REPOSITORY=https://github.com/hong19860320/Paddle-Lite.git -DARM_DNN_LIBRARY_TAG=hongming/fix_arm_dnn_for_padle"
  fi
  build_dir=$PADDLE_DIR/build
  lib_root=$ROOT_DIR/libs/PaddleInference
  lib_dir=$lib_root/$os/$lib_abi
  mkdir -p $build_dir
  cd $build_dir
  if [ $rebuild -eq 1 ]; then
    rm -rf $build_dir/*
    $build_cmd $extra_args
  fi
  make -j8
  rm -rf $lib_dir/$device_name
  mkdir -p $lib_dir/$device_name
  cp -rf $build_dir/paddle_inference_install_dir/* $lib_dir/$device_name/
  echo "Done"
}

# os: android, linux
# arch: armv7, armv8, armv7hf, x86
# rebuild: 0, 1
# device_name: cpu/xpu

#build_and_update_lib linux armv8 1 cpu
#build_and_update_lib linux armv8 1 xpu
build_and_update_lib linux x86 1 cpu
#build_and_update_lib linux x86 1 xpu

echo "Done."
