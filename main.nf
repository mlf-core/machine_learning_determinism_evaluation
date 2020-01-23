process train_mnist_pytorch {
    echo true

    label (params.GPU == "ON" ? 'with_gpus': 'with_cpus')

    script:
    """
    train_mnist_pytorch.py
    """
}
