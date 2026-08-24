# # ============================================================================
# # mamba-ssm dev environment for RTX 5070 (Blackwell / sm_120)
# # Base: CUDA 12.9 devel image (matches the "cu129"-built accelerator wheels)
# # Conda env: stagm-env, Python 3.12 (required by the cp312 wheels below)
# # ============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN conda create -n stagm-env python=3.11 -y && conda clean -afy

SHELL ["conda", "run", "-n", "stagm-env", "/bin/bash", "-c"]

RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

RUN pip install --no-cache-dir torch==2.9.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir torch_geometric
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE
ENV MAMBA_FORCE_BUILD=TRUE
ENV MAMBA_SKIP_CUDA_BUILD=FALSE
ENV MAX_JOBS=4

RUN pip install --no-cache-dir causal-conv1d --no-build-isolation

RUN pip install --no-cache-dir mamba-ssm --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN python -m ipykernel install --name stagm-env --display-name "Python (stagm-env)"

RUN echo "conda activate stagm-env" >> /etc/bash.bashrc
ENV PATH=${CONDA_DIR}/envs/stagm-env/bin:${PATH}

WORKDIR /workspace
EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "stagm-env", \
    "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]