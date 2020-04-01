process train_mnist_pytorch {
    echo true
    container 'mlflowcore/pytorch:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.pytorch

    script:
    """
    train_mnist_pytorch.py --epochs ${params.epochs}
    """
}

process train_mnist_tensorflow {
    echo true
    container 'mlflowcore/tensorflow:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.tensorflow

    script:
    """
    train_mnist_tensorflow_custom.py
    """
}

process train_boston_xgboost {
    echo true
    container 'mlflowcore/xgboost:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.xgboost

    script:
    """
    train_boston_xgboost.py --epochs ${params.epochs}
    """
}
