FROM nvidia/cuda:10.1-base-ubuntu18.04
LABEL authors="Lukas Heumos (lukas.heumos@posteo.net)" \
      description="Docker image containing all requirements for running Pytorch on CUDA enabled GPUs"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

 # Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user 
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

 # Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Update Conda first
RUN conda update conda

# Install the conda environment
COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a

# Activate the environment (source: https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-481241498)
RUN echo "source activate pytorch-1.4-cuda-10.1" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Dump the details of the installed packages to a file for posterity
RUN cd /home/user && mkdir output && cd output && \
conda env export --name pytorch-1.4-cuda-10.1 > pytorch-1.4-cuda-10.1.yml
