ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Project paths
ARG LENEWTON_PATH_ARG=/workspace/lenewton
ARG DOCKER_USER_HOME_ARG=/root
ENV LENEWTON_PATH=${LENEWTON_PATH_ARG}
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}

# Workdir and default shell
WORKDIR ${LENEWTON_PATH}
SHELL ["/bin/bash", "-c"]

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluetooth \
    bluez \
    bluez-tools \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    freeglut3-dev \
    gcc \
    git \
    graphviz \
    ibverbs-providers \
    libffi-dev \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libhidapi-hidraw0 \
    libhidapi-libusb0 \
    libibverbs1 \
    libjpeg-dev \
    libosmesa6-dev \
    libpng-dev \
    librdmacm1 \
    libssl-dev \
    libosmesa6 \
    libosmesa6-dev \
    libx11-6 \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxi-dev \
    libxrandr-dev \
    mesa-utils \
    openjdk-8-jdk \
    openssh-client \
    patchelf \
    swig \
    unzip \
    vim \
    wget \
    xvfb \
    x11-apps \
    x11-xserver-utils \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install uv to system location
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="${DOCKER_USER_HOME}/.cargo/bin:$PATH"

# Set up default python environment
RUN echo 'alias python="uv run python"' >> ~/.bashrc

# Set working directory
WORKDIR ${LENEWTON_PATH}
