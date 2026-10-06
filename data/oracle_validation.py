"""Independent, bounded MNIST copies for the explicit Oracle CPU comparison."""
import copy


def copy_mnist_prefix(dataset, sample_count):
    """
    Keep a deterministic real-MNIST prefix with independent images and labels.
    :param dataset: Loaded torchvision MNIST dataset with transforms intact.
    :param sample_count: Positive number of source samples to retain.
    :return: Independent MNIST object containing the requested prefix.
    """
    if type(sample_count) is not int or not 0 < sample_count <= len(dataset):
        raise ValueError('MNIST sample count must fit the source dataset.')
    result = copy.copy(dataset)
    result.data = dataset.data[:sample_count].clone()
    result.targets = dataset.targets[:sample_count].clone()
    return result
