#!/bin/bash
docker build --network=host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy -t paddle-cpu-qemu-user-static-x86_64-aarch64-ubuntu-gcc82:18.04 .
