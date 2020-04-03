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
    //container 'tensorflow/tensorflow:2.2.0rc2-gpu-py3'
    container 'mlflowcore/tensorflow:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.tensorflow

    script:
    """
    train_mnist_tensorflow.py --epochs ${params.epochs}
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
