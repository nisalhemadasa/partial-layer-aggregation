"""
Description: This module defines Ditto's global aggregation strategy.

Ditto's personalized client training is added in a later integration phase. The
global path aggregates only the ordinary upload model using client sample counts.
"""
from collections import OrderedDict
import copy
import math
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

import constants
from device_utils import get_device
from models.utils import set_parameters


DITTO_DEFAULTS = {
    'ditto_lambda': 0.05,
    'ditto_learning_rate': None,
    'ditto_personal_epochs': None,
    'ditto_eval_personalized': True,
    'ditto_dynamic_lambda': False,
    'ditto_lambda_candidates': [0.1, 1.0, 2.0],
    'ditto_validation_fraction': 0.1,
    'ditto_validation_seed': 42,
}


def resolve_ditto_parameters(parameters: Dict) -> Dict:
    """
    Resolve and validate Ditto configuration values.
    :param parameters: Experiment recovery parameters.
    :return: Validated Ditto-specific parameters.
    """
    resolved = dict(DITTO_DEFAULTS)
    resolved.update({key: parameters[key] for key in DITTO_DEFAULTS if key in parameters})

    ditto_lambda = resolved['ditto_lambda']
    if (isinstance(ditto_lambda, bool) or not isinstance(ditto_lambda, (int, float)) or
            not math.isfinite(ditto_lambda) or ditto_lambda < 0):
        raise ValueError("ditto_lambda must be a finite non-negative number.")
    resolved['ditto_lambda'] = float(ditto_lambda)

    learning_rate = resolved['ditto_learning_rate']
    if learning_rate is not None:
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
                not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("ditto_learning_rate must be None or a finite positive number.")
        resolved['ditto_learning_rate'] = float(learning_rate)

    personal_epochs = resolved['ditto_personal_epochs']
    if personal_epochs is not None and (isinstance(personal_epochs, bool) or
                                        not isinstance(personal_epochs, int) or personal_epochs <= 0):
        raise ValueError("ditto_personal_epochs must be None or a positive integer.")

    for flag in ('ditto_eval_personalized', 'ditto_dynamic_lambda'):
        if not isinstance(resolved[flag], bool):
            raise ValueError(flag + " must be a boolean.")

    candidates = resolved['ditto_lambda_candidates']
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("ditto_lambda_candidates must be a non-empty list or tuple.")
    normalized_candidates = []
    for candidate in candidates:
        if (isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or
                not math.isfinite(candidate) or candidate < 0):
            raise ValueError("ditto_lambda_candidates must contain finite non-negative numbers.")
        normalized_candidates.append(float(candidate))
    resolved['ditto_lambda_candidates'] = normalized_candidates

    validation_fraction = resolved['ditto_validation_fraction']
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or
            not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("ditto_validation_fraction must be between zero and one.")
    resolved['ditto_validation_fraction'] = float(validation_fraction)

    validation_seed = resolved['ditto_validation_seed']
    if isinstance(validation_seed, bool) or not isinstance(validation_seed, int):
        raise ValueError("ditto_validation_seed must be an integer.")

    return resolved


def compute_ditto_proximal_loss(personal_model: nn.Module, global_model_params: OrderedDict,
                                ditto_lambda: float) -> torch.Tensor:
    """
    Compute Ditto's proximal penalty over trainable named parameters.
    :param personal_model: Persistent personalized client model.
    :param global_model_params: Detached round-start server state.
    :param ditto_lambda: Non-negative proximal regularization strength.
    :return: Scalar proximal-loss tensor.
    """
    if ditto_lambda < 0:
        raise ValueError("ditto_lambda must be non-negative.")
    if global_model_params is None:
        raise ValueError("global_model_params is required for Ditto proximal training.")

    proximal_loss = None
    for parameter_name, personal_parameter in personal_model.named_parameters():
        if parameter_name not in global_model_params:
            raise KeyError("Global model parameters are missing Ditto parameter: " + parameter_name)
        global_parameter = global_model_params[parameter_name].detach().to(
            device=personal_parameter.device,
            dtype=personal_parameter.dtype
        )
        parameter_loss = torch.sum(torch.square(personal_parameter - global_parameter))
        proximal_loss = parameter_loss if proximal_loss is None else proximal_loss + parameter_loss

    if proximal_loss is None:
        proximal_loss = torch.tensor(0.0, device=get_device())
    return 0.5 * float(ditto_lambda) * proximal_loss


def evaluate_ditto_validation_loss(personal_model: nn.Module, validation_loader: DataLoader) -> float:
    """
    Evaluate mean NLL loss on a client's held-out Ditto validation partition.
    :param personal_model: Candidate personalized model.
    :param validation_loader: Persistent client-local validation loader.
    :return: Mean validation loss per sample.
    """
    if validation_loader is None:
        raise ValueError("Dynamic Ditto lambda selection requires a validation loader.")
    criterion = nn.NLLLoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0
    was_training = personal_model.training
    personal_model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(get_device(), non_blocking=True).float()
            labels = labels.to(get_device(), non_blocking=True).long()
            total_loss += criterion(personal_model(inputs), labels).item()
            total_samples += labels.size(0)
    if was_training:
        personal_model.train()
    if total_samples == 0:
        raise ValueError("Dynamic Ditto lambda validation partition is empty.")
    return total_loss / total_samples


def train_ditto_personal_model(personal_model: nn.Module, trainloader: DataLoader,
                               global_model_params: OrderedDict, ditto_lambda: float,
                               epochs: int, learning_rate: float = 0.01,
                               dynamic_lambda: bool = False,
                               lambda_candidates: List[float] = None,
                               validation_loader: DataLoader = None) -> List[Dict]:
    """
    Train a persistent Ditto personalized model against a fixed global reference.
    :param personal_model: Persistent personalized client model.
    :param trainloader: Client local-training loader.
    :param global_model_params: Detached round-start server state.
    :param ditto_lambda: Proximal regularization strength.
    :param epochs: Number of personalized local epochs.
    :param learning_rate: Personalized SGD learning rate.
    :param dynamic_lambda: Whether to choose lambda separately for every local batch.
    :param lambda_candidates: Ordered candidate values for dynamic selection.
    :param validation_loader: Persistent client validation loader.
    :return: Dynamic-lambda decision records; empty for fixed-lambda training.
    """
    if trainloader is None:
        raise ValueError("Ditto personalized training requires a trainloader.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Ditto personalized epochs must be a positive integer.")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Ditto personalized learning rate must be finite and positive.")
    if dynamic_lambda and (not lambda_candidates or validation_loader is None):
        raise ValueError("Dynamic Ditto lambda selection requires candidates and a validation loader.")

    personal_model.to(get_device()).float()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(personal_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    decisions = []

    for epoch_idx in range(epochs):
        personal_model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(get_device(), non_blocking=True).float()
            labels = labels.to(get_device(), non_blocking=True).long()

            if dynamic_lambda:
                base_optimizer_state = copy.deepcopy(optimizer.state_dict())
                candidate_results = []
                best_loss = None
                best_model_state = None
                best_optimizer_state = None
                selected_candidate_index = None

                for candidate_index, candidate_lambda in enumerate(lambda_candidates):
                    candidate_model = copy.deepcopy(personal_model).to(get_device())
                    candidate_optimizer = torch.optim.SGD(candidate_model.parameters(), lr=learning_rate,
                                                          momentum=0.9, weight_decay=5e-4)
                    candidate_optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))
                    candidate_optimizer.zero_grad(set_to_none=True)
                    outputs = candidate_model(inputs)
                    candidate_loss = criterion(outputs, labels)
                    candidate_loss = candidate_loss + compute_ditto_proximal_loss(
                        candidate_model, global_model_params, candidate_lambda)
                    candidate_loss.backward()
                    candidate_optimizer.step()
                    validation_loss = evaluate_ditto_validation_loss(candidate_model, validation_loader)
                    candidate_results.append({
                        'lambda': float(candidate_lambda),
                        'validation_loss': float(validation_loss)
                    })
                    # Strict comparison gives deterministic configured-order tie breaking.
                    if best_loss is None or validation_loss < best_loss:
                        best_loss = validation_loss
                        best_model_state = copy.deepcopy(candidate_model.state_dict())
                        best_optimizer_state = copy.deepcopy(candidate_optimizer.state_dict())
                        selected_candidate_index = candidate_index

                personal_model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
                # The selected candidate optimizer performed the real step. Mark the base optimizer accordingly so
                # its epoch scheduler advances without treating this as a scheduler-before-optimizer call.
                optimizer._opt_called = True
                decisions.append({
                    'epoch': epoch_idx,
                    'batch': batch_idx,
                    'candidates': candidate_results,
                    'selected_candidate_index': selected_candidate_index,
                    'selected_lambda': float(lambda_candidates[selected_candidate_index])
                })
            else:
                optimizer.zero_grad(set_to_none=True)
                outputs = personal_model(inputs)
                loss = criterion(outputs, labels)
                loss = loss + compute_ditto_proximal_loss(personal_model, global_model_params, ditto_lambda)
                loss.backward()
                optimizer.step()
        scheduler.step()

    return decisions


class Ditto:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, server_model: nn.Module,
                         client_model_params_dict: Dict[int, OrderedDict],
                         client_sample_counts_dict: Dict[int, int]) -> None:
        """
        Aggregate Ditto upload models using local-training sample counts.
        :param server_model: Server model updated by the aggregate.
        :param client_model_params_dict: Upload-model state dictionaries keyed by client ID.
        :param client_sample_counts_dict: Positive local-training sample counts keyed by client ID.
        :return: None.
        """
        if not client_model_params_dict:
            raise ValueError("Ditto aggregation requires at least one client upload.")
        if not client_sample_counts_dict:
            raise ValueError("Ditto aggregation requires client sample counts.")

        client_ids = list(client_model_params_dict.keys())
        if set(client_ids) != set(client_sample_counts_dict.keys()):
            raise ValueError("Ditto client uploads and sample counts must have identical client IDs.")

        sample_counts = {}
        for client_id in client_ids:
            sample_count = client_sample_counts_dict[client_id]
            if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
                raise ValueError("Ditto client sample counts must be positive integers.")
            sample_counts[client_id] = sample_count

        server_parameters = server_model.state_dict()
        expected_keys = list(server_parameters.keys())
        for client_id, parameters in client_model_params_dict.items():
            if list(parameters.keys()) != expected_keys:
                raise ValueError("Ditto client model keys do not match the server model for client "
                                 + str(client_id) + ".")
            for key in expected_keys:
                if parameters[key].shape != server_parameters[key].shape:
                    raise ValueError("Ditto tensor shape does not match the server model for client "
                                     + str(client_id) + " and key " + key + ".")

        total_samples = sum(sample_counts.values())
        largest_client_id = max(client_ids, key=lambda client_id: sample_counts[client_id])
        aggregated_parameters = OrderedDict()

        with torch.no_grad():
            for key, server_parameter in server_parameters.items():
                if torch.is_floating_point(server_parameter) or torch.is_complex(server_parameter):
                    accumulator = torch.zeros_like(server_parameter)
                    for client_id in client_ids:
                        client_parameter = client_model_params_dict[client_id][key].detach().to(
                            device=server_parameter.device,
                            dtype=server_parameter.dtype
                        )
                        accumulator.add_(client_parameter, alpha=sample_counts[client_id] / total_samples)
                    aggregated_parameters[key] = accumulator
                else:
                    # Integer/bool buffers cannot be averaged safely. Keep the largest contributor's value.
                    aggregated_parameters[key] = client_model_params_dict[largest_client_id][key].detach().to(
                        device=server_parameter.device,
                        dtype=server_parameter.dtype
                    ).clone()

        set_parameters(server_model, aggregated_parameters)


def aggregator_fn():
    """Return a Ditto aggregation strategy."""
    return Ditto(strategy_name=constants.RecoveryAlgorithm.DITTO)
