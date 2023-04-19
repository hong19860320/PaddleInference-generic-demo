#!/bin/bash
set -e

source settings.sh

build_and_update_lib() {
  local os=$1
  local arch=$2
  local rebuild=$3
  local device_name=$4
  local build_threads=$5
  if [ "$device_name" = "" ]; then
    echo "device_name is empty!"
    return -1
  fi
  if [ "$build_threads" = "" ]; then
    build_threads=$(nproc)
  fi
  cmake_args="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=\"-Wno-error -w\" -DPY_VERSION=3.7 -DPYTHON_EXECUTABLE=`which python3` -DON_INFER=ON -DWITH_TESTING=OFF -DWITH_XBYAK=OFF -DWITH_NCCL=OFF -DWITH_DISTRIBUTE=OFF -DWITH_PYTHON=OFF"
  make_args=""
  if [ "$os" = "android" ]; then
    # Android
    cmake_args="$cmake_args -DWITH_MKL=OFF -DWITH_GPU=OFF"
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
    # Linux
    if [ "$arch" = "armv8" ]; then
      lib_abi="arm64"
      cmake_args="$cmake_args -DWITH_MKL=OFF -DWITH_GPU=OFF"
      make_args="$make_args TARGET=ARMV8"
      if [ "$device_name" == "xpu" ]; then
        cmake_args="$cmake_args -DWITH_XPU=ON"
      fi
    elif [ "$arch" = "armv7hf" ]; then
      lib_abi="armhf"
      cmake_args="$cmake_args -DWITH_MKL=OFF -DWITH_GPU=OFF"
    elif [ "$arch" = "x86" ]; then
      lib_abi="amd64"
      cmake_args="$cmake_args -DWITH_MKL=ON -DWITH_GPU=OFF"
      if [ "$device_name" == "xpu" ]; then
        cmake_args="$cmake_args -DWITH_XPU=ON"
      fi
    else
      echo "Abi $arch is not supported for $os and any devices."
      return -1
    fi
    lib_os="linux"
  else
    # QNX
    cmake_args="$cmake_args -DWITH_MKL=OFF -DWITH_GPU=OFF"
    if [ "$arch" = "armv8" ]; then
      lib_abi="arm64"
    else
      echo "Abi $arch is not supported for $os and any devices."
      return -1
    fi
    lib_os="qnx"
  fi
  if [ "$arch" = "armv8" ] || [ "$arch" = "armv7hf" ]; then
    #cmake_args="$cmake_args -DWITH_ARM=ON -DWITH_ARM_DNN_LIBRARY=ON -DARM_DNN_LIBRARY_REPOSITORY=https://github.com/hong19860320/Paddle-Lite.git -DARM_DNN_LIBRARY_TAG=hongming/fix_arm_dnn_for_padle"
    cmake_args="$cmake_args -DWITH_ARM=ON"
  fi
  build_dir=$PADDLE_DIR/build.$os.$arch.$device_name
  lib_root=$ROOT_DIR/libs/PaddleInference
  lib_dir=$lib_root/$os/$lib_abi
  mkdir -p $build_dir
  cd $build_dir
  if [ $rebuild -eq 1 ]; then
    rm -rf $build_dir/*
    echo "cmake .. $cmake_args"
    cmake .. $cmake_args
  fi
  echo "make $make_args -j$build_threads"
  make $make_args -j$build_threads
  rm -rf $lib_dir/$device_name
  mkdir -p $lib_dir/$device_name
  cp -rf $build_dir/paddle_inference_install_dir/* $lib_dir/$device_name/
  echo "Done"
}

# os: android, linux
# arch: armv7, armv8, armv7hf, x86
# rebuild: 0, 1
# device_name: cpu/xpu
# build_threads: 8/16/32/64

#build_and_update_lib linux armv8 1 cpu
#build_and_update_lib linux armv8 1 xpu
#build_and_update_lib linux x86 1 cpu
build_and_update_lib linux x86 1 xpu
echo "Done."
