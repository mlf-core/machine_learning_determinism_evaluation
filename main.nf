process train_mnist_pytorch {
    echo true
    container 'mlflowcore/pytorch:dev'

    switch (params.platform) {
        case 'all_gpu': 
        label 'with_all_gpus'
        case 'single_gpu':
        label 'with_single_gpu' 
        case 'cpu': 
        label 'with_cpus'
    }

    when: params.pytorch

    script:
    """
    train_mnist_pytorch.py --epochs ${params.epochs}
    """
}

process train_mnist_tensorflow {
    echo true
    container 'mlflowcore/tensorflow:dev'

    switch (params.platform) {
        case 'all_gpu': 
        label 'with_all_gpus'
        case 'single_gpu':
        label 'with_single_gpu' 
        case 'cpu':
        label 'with_cpus'
    }

    when: params.tensorflow

    script:
    """
    train_mnist_tensorflow.py --epochs ${params.epochs}
    """
}

process train_boston_covtype_xgboost {
    echo true
    container 'mlflowcore/xgboost:dev'

    switch (params.platform) {
        case 'all_gpu': 
        label 'with_all_gpus'
        case 'single_gpu':
        label 'with_single_gpu' 
        case 'cpu':
        label 'with_cpus'
    }

    when: params.xgboost
    when: !params.platform == 'all_gpu'

    script:
    """
    train_boston_covtype_xgboost.py --epochs ${params.epochs} --dataset ${params.dataset} --no-cuda ${params.no_cuda}
    """
}

process train_boston_covtype_dask_xgboost {
    echo true
    container 'mlflowcore/xgboost:dev'

    switch (params.platform) {
        case 'all_gpu': 
        label 'with_all_gpus'
        case 'single_gpu':
        label 'with_single_gpu' 
        case 'cpu':
        label 'with_cpus'
    }

    when: params.xgboost
    when: params.dask

    script:
    """
    train_boston_covtype_dask_xgboost.py --epochs ${params.epochs} --dataset ${params.dataset} --n-gpus ${params.n_gpus}
    """
}
