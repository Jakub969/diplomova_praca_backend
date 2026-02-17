FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=6.1

WORKDIR /workspace

# System deps
RUN apt update && apt install -y \
    ninja-build \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git

# Python deps
RUN pip install --no-cache-dir \
    ninja \
    open3d \
    scikit-learn \
    numpy

# Copy project
COPY . .

# Build PointNet++ CUDA ops
WORKDIR /workspace/Pointnet2_PyTorch/pointnet2_ops_lib
RUN python setup.py install

WORKDIR /workspace
CMD ["/bin/bash"]
