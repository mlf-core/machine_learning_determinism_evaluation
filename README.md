# nextflow_pytorch_gpu

## Proof of concept for running deep learning on GPUs using nextflow
Training a simple 2d convolutional neural network (2d conv, 2d conv, dropout (0.25), dropout (0.5), fc, fc) on MNIST on the GPU.

## Requirements

* Docker
* Cuda
* Nvidia-container-toolkit
* A Cuda enabled GPU

## Run 

nextflow run main.nf --GPU YES -with-docker

Note, that the label possibly has to be set manually. There may be some nasty hidden bug (https://github.com/nextflow-io/nextflow/issues/1471)
