#!/bin/bash
docker run --name test-paddle-cpu-x86_64 -it \
  --privileged --pids-limit 409600 --network=host \
  -v $PWD:/Work -w /Work \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  registry.baidubce.com/paddlepaddle/paddle:latest-dev \
  /bin/bash
