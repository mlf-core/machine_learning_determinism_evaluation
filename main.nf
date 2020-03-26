/* process train_mnist_pytorch {
    echo true
    container 'ml/pytorch:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    script:
    """
    train_mnist_pytorch.py --epochs ${params.epochs}
    """

    when: params.pytorch
} */

process train_mnist_tensorflow {
    echo true
    container 'ml/tensorflow:dev'

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    script:
    """
    train_mnist_tensorflow.py
    """

}
