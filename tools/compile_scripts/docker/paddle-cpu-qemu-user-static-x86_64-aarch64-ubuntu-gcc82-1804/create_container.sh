#!/bin/bash
docker run --name test-paddle-cpu-qemu-x86_64-aarch64 -it \
  --privileged --pids-limit 409600 --network=host \
  -v $PWD:/Work -w /Work \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  paddle-cpu-qemu-user-static-x86_64-aarch64-ubuntu-gcc82:18.04 \
  /bin/bash
