"""
Description: This constants used in this project

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""


# Directory paths related to datasets
class Paths:
    # path to download and read the datasets
    DATASET = 'data/'

    # path related to saved plots
    PLOT_SAVE_PATH = './plots/saved_plots/'

    # path related to saved logs
    LOG_SAVE_PATH = './logs/saved_logs/'


# File extensions
class FileExtesions:
    PKL = '.pkl'
    CSV = '.csv'


# String related to miscellaneous messages
class MiscMessages:
    ACCURACY = "accuracy"


# Names of the dataset
class DatasetNames:
    MNIST = 'MNIST'
    F_MNIST = 'FashionMNIST'
    CIFAR_10 = 'cifar-10-batches-py'


class DatasetFileNames:
    # MNIST dataset file names
    MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    # F_MNIST dataset file names
    F_MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    F_MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    F_MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    F_MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    def get_train_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_IMAGES]
        else:
            pass

    def get_train_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_LABELS]
        else:
            pass

    def get_test_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_IMAGES]
        else:
            pass

    def get_test_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_LABELS]
        else:
            pass


# Strings related to plots
class Plots:
    NUMBER_OF_ROUNDS = 'Rounds'
    LOSS = 'Loss'
    ACCURACY = 'Accuracy'
    # for client individual loss/accuracy vs rounds plot
    CLIENT_LOSS_VS_ROUNDS_TITLE = 'Loss per Client Across Rounds'
    CLIENT_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per Client Across Rounds'
    CLIENT_LOSS_VS_ROUNDS_PNG = 'client_loss_vs_rounds'
    CLIENT_ACCURACY_VS_ROUNDS_PNG = 'client_accuracy_vs_rounds'
    # for server individual loss/accuracy vs rounds plot
    SERVER_LOSS_VS_ROUNDS_TITLE = 'Loss per servers Across Rounds'
    SERVER_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per server Across Rounds'
    SERVER_LOSS_VS_ROUNDS_PNG = 'server_loss_vs_rounds'
    SERVER_ACCURACY_VS_ROUNDS_PNG = 'server_accuracy_vs_rounds'
    # for client level average loss/accuracy vs rounds plot
    CLIENT_AVG_LOSS_VS_ROUNDS_TITLE = 'Client Level Average Loss Across Rounds'
    CLIENT_AVG_LOSS_VS_ROUNDS_PNG = 'client_avg_loss_vs_rounds'
    CLIENT_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Client Level Average Accuracy Across Rounds'
    CLIENT_AVG_ACCURACY_VS_ROUNDS_PNG = 'client_avg_accuracy_vs_rounds'
    # for server level average loss/accuracy vs rounds plot
    SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_TITLE = 'Server Level Average Loss Across Rounds'
    SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_PNG = 'server_level_avg_loss_vs_rounds'
    SERVER_OVERALL_AVG_LOSS_VS_ROUNDS_TITLE = 'Server Overall Average Loss Across Rounds'
    SERVEr_OVERALL_AVG_LOSS_VS_ROUNDS_PNG = 'server_overall_avg_loss_vs_rounds'
    SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Server Level Average Accuracy Across Rounds'
    SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_PNG = 'server_level_avg_accuracy_vs_rounds'
    SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Server Overall Average Accuracy Across Rounds'
    SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_PNG = 'server_overall_avg_accuracy_vs_rounds'
    # for client-edge server distance plot
    L2_DISTANCE = 'L2 Distance'
    CLIENT_EDGE_SERVER_DISTANCE_VS_ROUNDS_TITLE = 'Client-Edge Server Distance Across Rounds'
    CLIENT_EDGE_SERVER_DISTANCE_VS_ROUNDS_PNG = 'client_edge_server_distance_vs_rounds'
    # for client-edge-layer server distance plot
    CLIENT_SERVER_LAYER_DISTANCE_VS_ROUNDS_TITLE = 'Client-Server-Layer Distance Across Rounds'
    CLIENT_SERVER_LAYER_DISTANCE_VS_ROUNDS_PNG = 'client_server_layer_distance_vs_rounds'
    # label distribution plots for iid and non-iid client data distribution
    LABELS = 'Labels'
    SAMPLES_COUNT = 'Samples Count'
    LABEL_DISTRIBUTION_TITLE = 'Label Distribution per Client'
    LABEL_DISTRIBUTION_PNG = 'label_distribution_per_client'

# Drift patterns
class DriftPatterns:
    ABRUPT = 'abrupt'
    GRADUAL = 'gradual'
    INCREMENTAL = 'incremental'
    REOCCURRING = 'reoccurring'
    INCREMENTAL_ABRUPT = 'incre-abrupt'
    ABRUPT_REOCURRING = 'abrupt-reoc'
    INCREMENTAL_REOCCURRING = 'incr-reoc'
    OUT_OF_CONTROL = 'out-of-control'


# Drift creation methods
class DriftCreationMethods:
    LABEL_SWAPPING = 'label_swapping'
    ROTATION = 'rotation'


# file names related to logging
class Logs:
    CLIENT_LOG = 'client_log'
    SERVER_LOG = 'server_log'
    CLIENT_LOG = 'non_drifted_client_log'
    CLIENT_AVG_LOG = 'client_avg_log'
    SERVER_LVL_AVG_LOG = 'server_level_avg_log'
    SERVER_OVERALL_AVG_LOG = 'server_overall_avg_log'

# Drift recovery algorithms
class RecoveryAlgorithm:
    # Client-side algorithms
    FEDAVG = 'fedavg'   # Adam-based training
    RRT = 'rrt'
    FEDAU = 'fedau'
    FLUID = 'fluid'
    # Server-side algorithms
    FEDAVG = 'fedavg'
