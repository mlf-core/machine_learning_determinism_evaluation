# Nextflow Machine Learning

## Proof of concept for running deep learning on GPUs using nextflow

Training a simple 2d convolutional neural network (2d conv, 2d conv, dropout (0.25), dropout (0.5), fc, fc) implemented in Pytorch on MNIST on the CPU or GPU.

## Requirements

* Docker
* Cuda
* Nvidia-container-toolkit
* A Cuda enabled GPU
* Nextflow
* openjdk 8 < x < 12

## Building the docker images locally

```bash
docker build -f Dockerfile_pytorch -t ml/pytorch:dev
```

## Running

Running with docker:
```bash
nextflow run main.nf --GPU ON -with-docker
```

Alternative you can use singularity to train your model:
```bash
nextflow run main.nf --GPU ON -with-singularity
```

Omit the parameter `--GPU ON` to train on the CPU.
