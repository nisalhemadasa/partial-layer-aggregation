"""
Description: This module contains the functions to plot the performance metrics of the federated network.

Author: Nisal Hemadasa
Date: 29-04-2025
Version: 1.0
"""
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")   # headless tkinter for linux server environments

# matplotlib.use('TkAgg')  # Or 'TkAgg', 'Qt5Agg', etc. Use this in windowed environments
import matplotlib.pyplot as plt
from numpy import sort

# Choose a backend based on what works best for your environment
from torch import Tensor

import constants
from data.dataset_loader import get_dataset_classnames_from_indices
from federated_network.client import Client


def plot_client_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]], file_save_path=None) -> None:
    """
    Plot the loss of the models against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the loss and accuracy of the models for each client
    :param file_save_path: Path to save the plot
    :return: None
    """
    # List to store the loss and accuracy values of all clients across all rounds
    all_client_losses = []
    all_clients_accuracies = []

    # Indicate the indices of the loss value and accuracy value in the tuple
    LOSS_INDEX = 0
    ACCURACY_INDEX = 1

    # Get the total number of clients to iterate
    num_clients = len(loss_and_accuracy[0])

    # Collect losses for each client across all rounds. Note: Here client_id = index of the client in the list
    for client_id in range(num_clients):
        client_losses = []
        client_accuracies = []
        for round_index in range(len(loss_and_accuracy)):
            client_losses.append(loss_and_accuracy[round_index][client_id][LOSS_INDEX])
            client_accuracies.append(loss_and_accuracy[round_index][client_id][ACCURACY_INDEX])
        all_client_losses.append(client_losses)
        all_clients_accuracies.append(client_accuracies)

    # Plot the loss of each client against the number of rounds
    plt.figure()  # Create a new figure for loss
    for client_id, losses in enumerate(all_client_losses):
        plt.plot(losses, label=f'Client {client_id}')

    if file_save_path is None:
        file_save_path = constants.Paths.PLOT_SAVE_PATH

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.CLIENT_LOSS_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_LOSS_VS_ROUNDS_PNG)

    # Plot the accuracy of each client against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for client_id, accuracies in enumerate(all_clients_accuracies):
        plt.plot(accuracies, label=f'Client {client_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.CLIENT_ACCURACY_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_ACCURACY_VS_ROUNDS_PNG)


def plot_server_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]], file_save_path=None) -> None:
    """
    Plot the loss and accuracy of the server model against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the loss and accuracy of the models for each client
    :param file_save_path: Path to save the plot
    :return: None
    """
    # List to store the loss and accuracy values of all clients across all rounds
    all_server_losses = []
    all_server_accuracies = []

    # Indicate the indices of the loss value and accuracy value in the tuple
    LOSS_INDEX = 0
    ACCURACY_INDEX = 1

    num_levels = len(loss_and_accuracy[0])

    # Collect losses for each client across all rounds. Note: Here client_id = index of the client in the list
    for level in range(num_levels):
        level_server_losses = []
        level_server_accuracies = []
        for server_id in range(len(loss_and_accuracy[0][level])):
            server_losses = []
            server_accuracies = []
            for round_index in range(len(loss_and_accuracy)):
                server_losses.append(loss_and_accuracy[round_index][level][server_id][LOSS_INDEX])
                server_accuracies.append(loss_and_accuracy[round_index][level][server_id][ACCURACY_INDEX])
            level_server_losses.append(server_losses)
            level_server_accuracies.append(server_accuracies)
        all_server_losses.append(level_server_losses)
        all_server_accuracies.append(level_server_accuracies)

    # Plot the loss of each server against the number of rounds
    plt.figure()  # Create a new figure for loss
    for level in range(num_levels):
        for server_id, losses in enumerate(all_server_losses[level]):
            plt.plot(losses, label=f'Level {level} Server {server_id}')

    if file_save_path is None:
        file_save_path = constants.Paths.PLOT_SAVE_PATH

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_LOSS_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVER_LOSS_VS_ROUNDS_PNG)

    # Plot the accuracy of each client against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for level in range(num_levels):
        for server_id, accuracies in enumerate(all_server_accuracies[level]):
            plt.plot(accuracies, label=f'Level {level} Server {server_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_ACCURACY_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVER_ACCURACY_VS_ROUNDS_PNG)


def plot_client_avg_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]], is_synchronous: bool,
                                          file_save_path=None) -> None:
    """
    Plot the average loss and accuracy of the clients against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the all client models for each
    round. - Outer List: drifted and non-drifted clients
           - Inner List: List of performance for each epoch
    :param is_synchronous: Boolean indicating if the drift synchronous or asynchronous
    :param file_save_path: Path to save the plot
    :return: None
    """
    # Extract the loss and accuracy values from the list of tuples
    non_drifted_client_avg_accuracies = [x[0] for x in loss_and_accuracy[0]]
    non_drifted_client_avg_losses = [x[1] for x in loss_and_accuracy[0]]

    drifted_client_avg_accuracies = [x[0] for x in loss_and_accuracy[1]]
    drifted_client_avg_losses = [x[1] for x in loss_and_accuracy[1]]

    # Plot the average loss of the clients against the number of rounds
    plt.figure()  # Create a new figure for loss
    plt.plot(non_drifted_client_avg_losses, label='Average Client Loss')
    plt.plot(drifted_client_avg_losses, label='Average Drifted Client Loss')

    if file_save_path is None:
        file_save_path = constants.Paths.PLOT_SAVE_PATH

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.CLIENT_AVG_LOSS_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the clients against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    plt.plot(non_drifted_client_avg_accuracies, label='Average Client Accuracy')
    plt.plot(drifted_client_avg_accuracies, label='Average Drifted Client Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.CLIENT_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_AVG_ACCURACY_VS_ROUNDS_PNG)


def plot_server_lvl_avg_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]], file_save_path=None) -> None:
    """
    Plot the average loss and accuracy of the server models for each level against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the models for each round of
    all server levels.  - List index: epochs
                        - Tuple index 0: depth level (root, ..., leaf)
    :param file_save_path: Path to save the plot
    :return: None
    """
    if not loss_and_accuracy:
        print('No data to plot')
        return

    num_level = len(loss_and_accuracy[0])  # Number of server tree hierarchy levels
    server_avg_accuracies = []  # stores accuracies for each level [root, ..., leaf]
    server_avg_losses = []  # stores losses for each level [root, ..., leaf]

    for level in range(num_level):
        server_avg_accuracies.append([x[level][1] for x in loss_and_accuracy])
        server_avg_losses.append([x[level][0] for x in loss_and_accuracy])

    # Plot the average loss of the server against the number of rounds
    plt.figure()  # Create a new figure for loss
    for level in range(num_level):
        plt.plot(server_avg_losses[level], label=f'Level {level} Average Server Loss')

    if file_save_path is None:
        file_save_path = constants.Paths.PLOT_SAVE_PATH

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the server against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for level in range(num_level):
        plt.plot(server_avg_accuracies[level], label=f'Level {level} Average Server Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_PNG)


def plot_server_overall_avg_performance_vs_rounds(loss_and_accuracy: List[Tuple], file_save_path=None) -> None:
    """
    Plot the average loss and accuracy of the total server against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the models for each round of
    all server
    :param file_save_path: Path to save the plot
    :return: None
    """
    # Extract the loss and accuracy values from the list of tuples
    server_avg_losses = [x[0] for x in loss_and_accuracy]
    server_avg_accuracies = [x[1] for x in loss_and_accuracy]

    # Plot the average loss of the server against the number of rounds
    plt.figure()  # Create a new figure for loss
    plt.plot(server_avg_losses, label='Average Server Loss')

    if file_save_path is None:
        file_save_path = constants.Paths.PLOT_SAVE_PATH

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_OVERALL_AVG_LOSS_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVEr_OVERALL_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the server against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    plt.plot(server_avg_accuracies, label='Average Server Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_PNG)


def plot_client_distance_vs_rounds(client_model_distance: List[Dict[int, float]],
                                   file_save_path: str = None) -> None:
    """
    Plot the L2 distance of the client-model-weights from the edge-server-model-weights against the number of training
    rounds
    :param client_model_distance: List of dictionaries containing the L2 distance of the client
    :param file_save_path: Path to save the plot
    :return : None
    """
    # Arrange the data for plotting in the structure {'key': [distances]}
    _client_distances = defaultdict(list)  # getting the correct empty structure
    for rnd in client_model_distance:
        for client_id, distance in rnd.items():
            _client_distances[client_id].append(distance)

    # Plot the client-edge server distance against the number of rounds
    plt.figure()  # Create a new figure
    for client_id, distance in _client_distances.items():
        plt.plot(distance, label=f'Client {client_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.L2_DISTANCE,
                            constants.Plots.CLIENT_EDGE_SERVER_DISTANCE_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_EDGE_SERVER_DISTANCE_VS_ROUNDS_PNG)


def plot_client_layer_distance_vs_rounds(client_layer_distance: List[Dict[int, Dict[str, Tensor]]],
                                         file_save_path: str = None) -> None:
    """
    Plot the L2 distance of the individual layers of the client-model from the corresponding layer of the
    edge-server-model-weights against the number of training rounds
    :param client_layer_distance:  List of dictionaries containing the L2 distance of the client model weights from
    the edge-server model weights for each layer
                                    - Dict key: layer name
                                    - Dict value: L2 distance
    :param file_save_path: Path to save the plot
    :return: None
    """
    # Initialize the new structure. Lambda handles the nested dictionary automatically
    _layer_distances = defaultdict(lambda: defaultdict(list))

    for rnd in client_layer_distance:
        for client_id, distance_dict in rnd.items():
            for key, distance in distance_dict.items():
                _layer_distances[client_id][key].append(distance)

    # Plot the client-edge server distance against the number of rounds
    plt.figure()  # Create a new figure
    for client_id, distance_dict in _layer_distances.items():
        for key, distance in distance_dict.items():
            plt.plot(distance, label=f'Client {client_id} Layer {key}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.L2_DISTANCE,
                            constants.Plots.CLIENT_SERVER_LAYER_DISTANCE_VS_ROUNDS_TITLE,
                            file_save_path + constants.Plots.CLIENT_SERVER_LAYER_DISTANCE_VS_ROUNDS_PNG)


def configure_and_save_plot(_plt, _x_label, _y_label, _title, _file_path, _legend_handles=None,
                            _label_rotate=None, _if_grid=None) -> None:
    """
    Add labels, title, legend and save the plot and displays it.
    :param _plt: The matplotlib pyplot object.
    :param _x_label: The label for the x - axis.
    :param _y_label: The label for the y - axis.
    :param _title: The title of the plot.
    :param _file_path: The path and file name to save the plot.
    :param _legend_handles: The legend handles to be displayed.
    :param _label_rotate: Rotation angle for x-axis labels.
    :param _if_grid: Boolean to indicate whether to show grid or not.
    :return: None
    """
    _plt.xlabel(_x_label)
    _plt.ylabel(_y_label)
    # _plt.title(_title)

    if _legend_handles is not None:
        _plt.legend(handles=_legend_handles, loc='best', fontsize=4)

    # if _if_grid is not None:
    #     _plt.grid()

    if _label_rotate is not None:
        _plt.xticks(rotation=_label_rotate, ha='right')  # test is horizontally aligned to right
        _plt.tight_layout()  # Automatically adjusts subplot margins so labels fit inside the figure.

    # Save the plot as a high-quality PNG
    png_path = f"{_file_path}.png"
    _plt.savefig(png_path, dpi=300)  # Increase DPI for higher resolution

    # Save the plot as a PDF
    pdf_path = f"{_file_path}.pdf"
    _plt.savefig(pdf_path, format="pdf")
    #
    # # Save PGF (LaTeX-friendly vector format)
    # pgf_path = f"{_file_path}.pgf"
    # _plt.savefig(pgf_path, format="pgf")

    # Display the plot
    # _plt.show()
    _plt.close()


def plot_dataset_distribution(clients_list: List[Client], _dataset_name: str,
                              file_save_path: str = None):
    """
    Create and save bar chart for the label counts of the train datasets of a given list of clients.
    :param clients_list: List of Client instances whose dataset distribution needs to be plotted.
    :param _dataset_name: Name of the dataset.
    :param file_save_path: Path to save the plot
    :return: None
    """
    for client in clients_list:
        sample_labels = [sample[1] for sample in client.local_trainset]  # Extract labels from the dataset
        distribution = Counter(sample_labels)  # Get the number of samples per label

        label_indices, sample_count = list(distribution.keys()), list(distribution.values())
        label_indices.sort()

        # Get the dataset class names
        labels = get_dataset_classnames_from_indices(_dataset_name, label_indices)

        # plot bar chart using label (x-axis labels) and count the (y-axis values)
        plt.figure()  # Create a new figure
        plt.bar(labels, sample_count, color='blue', edgecolor='black')

        configure_and_save_plot(plt, constants.Plots.LABELS, constants.Plots.SAMPLES_COUNT,
                                constants.Plots.LABEL_DISTRIBUTION_TITLE,
                                file_save_path + constants.Plots.LABEL_DISTRIBUTION_PNG + '_' + str(client.client_id),
                                _label_rotate=45)
