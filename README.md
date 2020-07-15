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
docker build -f Dockerfile_mlflowcore_base -t mlflowcore/base:1.0.0 .
docker build -f Dockerfile_pytorch -t mlflowcore/pytorch:dev .
docker build -f Dockerfile_tensorflow -t mlflowcore/tensorflow:dev .
docker build -f Dockerfile_xgboost -t mlflowcore/xgboost:dev .
```

## Running on your chosen platform
Choose one of: 
1. 'all_gpu' for all GPUs
2. 'single_gpu' for a single GPU
3. 'cpu' for running on CPU

Running with docker:
```bash
nextflow run main.nf --platform all_gpu/single_gpu/cpu -with-docker --tensorflow/pytorch/xgboost
```

Alternative you can use singularity to train your model:
```bash
nextflow run main.nf --platform all_gpu/single_gpu/cpu -with-singularity --tensorflow/pytorch/xgboost
```

Note that to run xgboost on the CPU you also need to use the parameter --no_cuda.

Running multiple GPUs on the local Dask cluster using XGBoost:
```bash
nextflow run main.nf -with-docker --platform all_gpu --xgboost --dataset boston --epochs 1000 --dask --n_gpus 2
```
