process train_mnist_pytorch {
    echo true
    container 'ml/pytorch:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.pytorch

    script:
    """
    train_mnist_pytorch.py --epochs ${params.epochs}
    """
}

process train_mnist_tensorflow {
    echo true
    container 'ml/tensorflow:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    when: params.tensorflow

    script:
    """
    train_mnist_tensorflow.py
    """
}
