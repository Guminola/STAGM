# ============================================================
# Base: NVIDIA CUDA 12.8 + cuDNN 9 on Ubuntu 24.04 LTS
# Chosen for: SM120/Blackwell support, cu128 PyTorch nightly,
# ============================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# ── Build-time args ─────────────────────────────────────────
ARG CONDA_ENV=stagm-env
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# Pinned to 2.8.0: latest stable with cu128 wheels AND matching
# PyG extension wheels on data.pyg.org.
ARG TORCH_VERSION=2.8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV=${CONDA_ENV}
ENV PATH=/opt/conda/bin:$PATH
# Forces SM_120 target for any extensions compiled from source
ENV TORCH_CUDA_ARCH_LIST="12.0"

# ── System packages ─────────────────────────────────────────
# build-essential provides gcc/g++ for PyG extension compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Install Miniforge ────────────────────────────────────────
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# ── Create conda environment: stagm-env ────────────────────
RUN conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} -c conda-forge -y \
    && conda clean -afy

# All subsequent RUN commands run inside the conda env
SHELL ["conda", "run", "--no-capture-output", "-n", "stagm-env", "/bin/bash", "-c"]

# ── Install PyTorch 2.8.0+cu128 ─────────────────────────────
RUN pip install --upgrade pip \
    && pip install torch==${TORCH_VERSION}+cu128 torchvision torchaudio \
    --index-url ${TORCH_INDEX}

# ── Install PyG (torch-geometric) + compiled extensions ─────
RUN pip install torch-geometric \
    && pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html \
    && ( pip install  pyg_lib -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html \
    || echo "pyg_lib wheel not available — skipping (optional)" )

# ── Install mamba-ssm ────────────────────────────────────────
RUN pip install mamba-ssm[causal-conv1d] --no-build-isolation

# ── User / project dependencies ─────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install  -r /tmp/requirements.txt

# ── Working directory ────────────────────────────────────────
WORKDIR /workspace

CMD ["conda", "run", "--no-capture-output", "-n", "stagm-env", "/bin/bash"]