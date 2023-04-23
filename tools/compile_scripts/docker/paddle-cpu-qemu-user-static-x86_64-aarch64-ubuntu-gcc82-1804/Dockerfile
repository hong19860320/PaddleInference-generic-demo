FROM multiarch/qemu-user-static:x86_64-aarch64 as qemu
FROM arm64v8/ubuntu:18.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

COPY --from=qemu /usr/bin/qemu-aarch64-static /usr/bin
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak \
    && sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list \
    && apt-get -y update

RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        gcc-8 \
        g++-8 \
        git \
        make \
        vim \
        wget \
        curl \
        fish \
        gawk \
        unzip \
        patch \
        less \
        openssh-server \
        libgomp1 \
        systemd \
        python3.7 \
        python3.7-distutils \
        openssh-client \
        sshpass

RUN ln -fs /usr/bin/gcc-8 /usr/bin/gcc
RUN ln -fs /usr/bin/g++-8 /usr/bin/g++
RUN ln -fs /usr/bin/python3.7 /usr/bin/python
RUN ln -fs /usr/bin/python3.7 /usr/bin/python3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install pyyaml

RUN curl -L https://cmake.org/files/v3.22/cmake-3.22.6-linux-aarch64.tar.gz -o - | tar -xz -C /opt/
RUN rm /usr/bin/cmake && ln -s /opt/cmake-3.22.6-linux-aarch64/bin/cmake /usr/bin/cmake && \
    rm /usr/bin/ccmake && ln -s /opt/cmake-3.22.6-linux-aarch64/bin/ccmake /usr/bin/ccmake

# timezone
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/timezone && echo 'Asia/Shanghai' >/etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata
