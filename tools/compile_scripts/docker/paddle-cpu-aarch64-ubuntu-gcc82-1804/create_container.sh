#!/bin/bash
docker run --name test-paddle-cpu-aarch64 -it \
  --privileged --pids-limit 409600 --network=host \
  -v $PWD:/Work -w /Work \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  registry.baidubce.com/device/paddle-cpu:ubuntu18-aarch64-gcc82 \
  /bin/bash
