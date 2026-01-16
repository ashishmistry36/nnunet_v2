FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Arguments for user
ARG XNAT_UNAME=${XNAT_UNAME}
ARG XNAT_GROUP=${XNAT_GROUP}
ARG XNAT_UID=${XNAT_UID}
ARG XNAT_GID=${XNAT_GID}

# Environment variables
ENV TZ=America/Chicago \
    PYTHONUNBUFFERED=1 \
    nnUNet_n_proc_DA=32 \
    nnUNet_raw="/tmp/raw" \
    nnUNet_preprocessed="/tmp/preprocessed" \
    nnUNet_results="/data/models" \
    RESULTS_FOLDER="/data/models" \
    VENV="/opt/venv" \
    UV_NO_CACHE="true" \
    PYTHON_VERSION="3.10" \
    UV_PYTHON_INSTALL_DIR="/usr/local/bin" \
    NUMEXPR_MAX_THREADS="32"

# System dependencies
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        sudo \
        tzdata \
        ffmpeg \
        libsm6 \
        libxext6 \
        libx11-6 \
        graphviz \
        cmake \
        build-essential \
        gcc \
        g++ \
        make \
        libgdcm-dev \
        libgdcm-tools && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# UV Python installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/bin/

# Create user and directories
RUN groupadd -g ${XNAT_GID} ${XNAT_GROUP} && \
    useradd -m -s /bin/bash -u ${XNAT_UID} -g ${XNAT_GID} -G sudo -l ${XNAT_UNAME} && \
    printf "\n\n%s ALL=(ALL) NOPASSWD:ALL\n\n" "${XNAT_UNAME}" >> /etc/sudoers && \
    mkdir -p ${nnUNet_raw} ${nnUNet_preprocessed} $(dirname ${VENV}) /app && \
    chown -R ${XNAT_UNAME}:${XNAT_GROUP} ${nnUNet_raw} ${nnUNet_preprocessed} $(dirname ${VENV}) /app

# Install Python
RUN uv python install ${PYTHON_VERSION}

USER ${XNAT_UID}
WORKDIR /app
ENV PATH="${VENV}/bin:$PATH"

# Create virtual environment
RUN uv venv --python ${PYTHON_VERSION} ${VENV}

# Install PyTorch 2.1.2 compatible with CUDA 11.8 (works with CUDA 11.4 runtime)
# Using --index-strategy to allow finding packages across multiple indexes
RUN uv pip install --python ${VENV}/bin/python --no-cache-dir \
        --index-strategy unsafe-best-match \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --extra-index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install remaining packages
COPY requirements.txt ./
RUN uv pip install --python ${VENV}/bin/python --no-cache-dir \
        --index-strategy unsafe-best-match \
        -r requirements.txt
