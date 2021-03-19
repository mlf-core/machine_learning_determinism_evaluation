#!/usr/bin/env bash
docker build -f Dockerfile_mlflowcore_base -t mlflowcore/base:1.0.0 .
docker build -f Dockerfile_pytorch -t mlflowcore/pytorch:dev .
docker build -f Dockerfile_tensorflow -t mlflowcore/tensorflow:dev .
docker build -f Dockerfile_xgboost -t mlflowcore/xgboost:dev .
