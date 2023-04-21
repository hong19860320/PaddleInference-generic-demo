#!/bin/bash
set -e

source build_and_update_libs.sh

build_and_update_lib linux x86 1 cpu
build_and_update_lib linux x86 1 xpu

echo "Done."
