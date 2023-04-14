#!/bin/bash
set -e

source settings.sh

build_and_update_lib() {
  local os=$1
  local arch=$2
  local rebuild_all=$3

  echo "Done"
}

# os: android, linux
# arch: armv7, armv8, armv7hf, x86
# rebuild_all: 0, 1

build_and_update_lib linux armv8 0

echo "Done."
