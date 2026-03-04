# ============================================================
# Base: NVIDIA CUDA 12.8 + cuDNN 9 on Ubuntu 22.04
# Chosen for: SM120/Blackwell support, cu128 PyTorch nightly,
# ============================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# ── Build-time args ─────────────────────────────────────────
ARG CONDA_ENV=stagam-env
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# Pinned to 2.8.0 — latest stable with cu128 wheels AND matching PyG extension wheels.
# Nightly (2.12.0.dev*) has no PyG extension wheels on data.pyg.org.
ARG TORCH_VERSION=2.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV=${CONDA_ENV}
ENV PATH=/opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="12.0"

# ── System packages ─────────────────────────────────────────
# Includes R build deps: libreadline, libblas, liblapack (needed by rpy2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential \
    libreadline-dev libcurl4-openssl-dev libssl-dev \
    libxml2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libblas-dev liblapack-dev \
    r-base r-base-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Install Miniforge ───────────────────────────────────────
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# ── Create conda environment: stagam-env ────────────────────
RUN conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} -c conda-forge -y \
    && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "stagam-env", "/bin/bash", "-c"]

# ── Install PyTorch stable 2.8.0+cu128 ──────────────────────
RUN pip install --upgrade pip \
    && pip install torch==${TORCH_VERSION}+cu128 torchvision torchaudio \
    --index-url ${TORCH_INDEX}

# ── Install PyG (torch-geometric) + extensions ──────────────
RUN pip install torch-geometric \
    && pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html \
    && pip install pyg_lib \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html \
    || echo "pyg_lib not available as wheel — skipping (torch-geometric works without it)"

# ── Install rpy2 ────────────────────────────────────────────
RUN pip install rpy2

# ── Install R package: mclust ───────────────────────────────
RUN R -e "install.packages('mclust', repos='https://cloud.r-project.org', quiet=TRUE)"

# ── Additional scientific Python packages ───────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --root-user-action=ignore -r /tmp/requirements.txt

# ── Set working directory ────────────────────────────────────
WORKDIR /workspace

# Default: launch bash inside the conda env
CMD ["conda", "run", "--no-capture-output", "-n", "stagam-env", "/bin/bash"]