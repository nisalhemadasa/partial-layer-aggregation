"""FedBABU strategy entry point.

The federated phase updates and aggregates only the model body. The shared
classifier head remains unchanged during aggregation.
"""

import constants
from models.utils import set_parameters
from strategy.FedBABU.utils import aggregate_fedbabu_body_parameters


class FedBABU:
    """Represent FedBABU in the framework's server strategy interface."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, server_model, client_model_params_dict) -> None:
        """
        Aggregate client bodies into the server model.

        :param server_model: Server model whose body will be updated.
        :param client_model_params_dict: Client model state dictionaries.
        :return: None
        """
        aggregated_body = aggregate_fedbabu_body_parameters(server_model, client_model_params_dict)
        set_parameters(server_model, aggregated_body, _strict=False)


def aggregator_fn() -> FedBABU:
    """
    Create the FedBABU strategy instance.

    :return: FedBABU strategy instance.
    """
    return FedBABU(strategy_name=constants.RecoveryAlgorithm.FEDBABU)
