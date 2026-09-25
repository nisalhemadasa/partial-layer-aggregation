"""Model partition helpers used by FedBABU."""

import copy
import math
from collections import OrderedDict
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

import constants
from models.utils import set_parameters, train
from device_utils import get_device


FEDBABU_HEAD_MODULE_BY_MODEL_TYPE = {
    constants.ModelTypes.CNN_MODEL: 'fc2',
    constants.ModelTypes.CNN_CIFAR_10: 'fc2',
    constants.ModelTypes.CNN_CIFAR_100: 'fc2',
    constants.ModelTypes.CONVNET_TINY_IMAGENET: 'head',
    # TabularAdultModel currently returns this literal from get_model_type().
    'TabularAdultModel': 'fc2',
}

FEDBABU_DEFAULTS = {
    'fedbabu_head_finetune_epochs': 5,
    'fedbabu_head_finetune_learning_rate': 0.01,
    'fedbabu_head_finetune_momentum': 0.5,
    'fedbabu_head_finetune_weight_decay': 0.0,
}


def resolve_fedbabu_parameters(parameters: Dict) -> Dict:
    """
    Resolve and validate the post-training FedBABU classifier settings.

    Defaults follow the official repository's fine-tuning epoch and base
    learning-rate defaults; optimizer details remain explicitly configurable.

    :param parameters: Experiment recovery parameters.
    :return: Validated FedBABU fine-tuning configuration.
    """
    if not isinstance(parameters, dict):
        raise ValueError("FedBABU parameters must be supplied as a dictionary.")
    resolved = dict(FEDBABU_DEFAULTS)
    resolved.update({key: parameters[key] for key in FEDBABU_DEFAULTS if key in parameters})

    epochs = resolved['fedbabu_head_finetune_epochs']
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("fedbabu_head_finetune_epochs must be a positive integer.")

    learning_rate = resolved['fedbabu_head_finetune_learning_rate']
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("fedbabu_head_finetune_learning_rate must be finite and positive.")
    resolved['fedbabu_head_finetune_learning_rate'] = float(learning_rate)

    momentum = resolved['fedbabu_head_finetune_momentum']
    if (isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or
            not math.isfinite(momentum) or not 0 <= momentum < 1):
        raise ValueError("fedbabu_head_finetune_momentum must be finite and in [0, 1).")
    resolved['fedbabu_head_finetune_momentum'] = float(momentum)

    weight_decay = resolved['fedbabu_head_finetune_weight_decay']
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or
            not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("fedbabu_head_finetune_weight_decay must be finite and non-negative.")
    resolved['fedbabu_head_finetune_weight_decay'] = float(weight_decay)
    return resolved


def train_fedbabu_body(model: nn.Module, dataset, epochs: int, verbose: bool = False) -> None:
    """
    Train the FedBABU model body through the framework's existing training routine.

    :param model: Supported model whose final linear classifier must remain fixed.
    :param dataset: Existing client training DataLoader.
    :param epochs: Number of local training epochs.
    :param verbose: Whether to print framework training progress.
    :return: None.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("FedBABU body training requires a positive integer epoch count.")
    body_state, _ = split_fedbabu_body_and_head(model)
    body_keys = set(body_state)
    parameters = dict(model.named_parameters())
    trainable_parameters = [parameter for name, parameter in parameters.items() if name in body_keys]
    if not trainable_parameters:
        raise ValueError("FedBABU model has no trainable body parameters.")

    original_requires_grad = {name: parameter.requires_grad for name, parameter in parameters.items()}
    try:
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in body_keys and original_requires_grad[name])
        if not any(parameter.requires_grad for parameter in trainable_parameters):
            raise ValueError("FedBABU model body has no parameters enabled for training.")
        train(model, dataset, epochs, verbose,
              _optimizer_parameters=[parameter for parameter in trainable_parameters
                                     if parameter.requires_grad])
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(original_requires_grad[name])


def train_fedbabu_head(model: nn.Module, dataset, parameters: Dict, verbose: bool = False) -> None:
    """
    Fine-tune only a FedBABU personal model's classifier on local training data.

    :param model: Per-client evaluation model initialized from the final server.
    :param dataset: DataLoader over this client's training data only.
    :param parameters: Resolved FedBABU fine-tuning settings.
    :param verbose: Whether to print framework training progress.
    :return: None.
    """
    if dataset is None or len(dataset) == 0:
        raise ValueError("FedBABU head fine-tuning requires non-empty local training data.")
    resolved = resolve_fedbabu_parameters(parameters)
    body_state, _ = split_fedbabu_body_and_head(model)
    head_name = FEDBABU_HEAD_MODULE_BY_MODEL_TYPE[model.get_model_type()]
    named_parameters = dict(model.named_parameters())
    head_prefix = head_name + '.'
    head_parameters = [parameter for name, parameter in named_parameters.items()
                       if name.startswith(head_prefix)]
    if not head_parameters:
        raise ValueError("FedBABU classifier has no trainable parameters.")

    body_snapshot = OrderedDict((key, value.detach().clone()) for key, value in body_state.items())
    original_requires_grad = {name: parameter.requires_grad for name, parameter in named_parameters.items()}
    modules = dict(model.named_modules())
    original_module_modes = {module: module.training for module in modules.values()}
    frozen_modules = [module for name, module in modules.items()
                      if name and name != head_name and not name.startswith(head_prefix) and
                      not head_name.startswith(name + '.')]
    try:
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name.startswith(head_prefix) and original_requires_grad[name])
        if not any(parameter.requires_grad for parameter in head_parameters):
            raise ValueError("FedBABU classifier parameters are disabled for training.")
        train(model, dataset, resolved['fedbabu_head_finetune_epochs'], verbose,
              _optimizer_parameters=[parameter for parameter in head_parameters if parameter.requires_grad],
              _learning_rate=resolved['fedbabu_head_finetune_learning_rate'],
              _momentum=resolved['fedbabu_head_finetune_momentum'],
              _weight_decay=resolved['fedbabu_head_finetune_weight_decay'],
              _frozen_modules=frozen_modules)
    finally:
        set_parameters(model, body_snapshot, _strict=False)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(original_requires_grad[name])
        for module, was_training in original_module_modes.items():
            module.training = was_training


def initialize_fedbabu_shared_head(clients, servers) -> None:
    """
    Copy one server's initial classifier head to every FedBABU model.

    :param clients: Client objects whose `model` heads should share the initial head.
    :param servers: Flat server list; the first server provides the canonical head.
    :return: None.
    """
    if not clients or not servers:
        raise ValueError("FedBABU shared-head initialization requires clients and servers.")
    if any(getattr(server, 'model', None) is None for server in servers):
        raise ValueError("FedBABU shared-head initialization requires a model on every server.")
    if any(getattr(client, 'model', None) is None for client in clients):
        raise ValueError("FedBABU shared-head initialization requires a model on every client.")

    canonical_body, canonical_head = split_fedbabu_body_and_head(servers[0].model)
    del canonical_body
    canonical_head = OrderedDict((key, value.detach().clone()) for key, value in canonical_head.items())
    destination_models = [server.model for server in servers] + [client.model for client in clients]
    for model in destination_models:
        _, model_head = split_fedbabu_body_and_head(model)
        if list(model_head) != list(canonical_head) or any(
                model_head[key].shape != canonical_head[key].shape for key in canonical_head):
            raise ValueError("FedBABU models must have identical classifier-head schemas.")

    for model in destination_models:
        set_parameters(model, canonical_head, _strict=False)


def initialize_fedbabu_personal_models(clients, servers) -> None:
    """
    Create one isolated post-training evaluation model per client.

    Each copy starts from the final flat server model, so it contains the final
    shared body and the same fixed classifier head. The models are retained on
    clients for the later personalized-head fine-tuning and evaluation steps.

    :param clients: Client objects receiving their own evaluation model copy.
    :param servers: Single-level FedBABU server list containing the final model.
    :return: None.
    """
    if not clients or len(servers) != 1:
        raise ValueError("FedBABU post-training models require clients and exactly one final server.")
    final_server_model = getattr(servers[0], 'model', None)
    if final_server_model is None:
        raise ValueError("FedBABU post-training model initialization requires a final server model.")
    final_state = final_server_model.state_dict()
    final_body, final_head = split_fedbabu_body_and_head(final_server_model)

    client_ids = [getattr(client, 'client_id', None) for client in clients]
    if any(client_id is None for client_id in client_ids) or len(set(client_ids)) != len(client_ids):
        raise ValueError("FedBABU post-training models require unique client IDs.")
    for client in clients:
        client_model = getattr(client, 'model', None)
        if client_model is None:
            raise ValueError("FedBABU post-training model initialization requires every client model.")
        client_body, client_head = split_fedbabu_body_and_head(client_model)
        if list(client_body) != list(final_body) or list(client_head) != list(final_head):
            raise ValueError("FedBABU client and final server model schemas must match.")

    for client in clients:
        personalized_model = copy.deepcopy(final_server_model).to(get_device())
        set_parameters(personalized_model, final_state, _strict=True)
        client.fedbabu_personal_model = personalized_model


def aggregate_fedbabu_tensor_values(values: Sequence[Tensor], reference_tensor: Tensor) -> Tensor:
    """
    Reduce client body tensors while preserving the server tensor's dtype.

    Floating-point and complex tensors use an equal mean. Non-floating buffers
    use an elementwise maximum so counters and flags remain valid discrete values.

    :param values: Corresponding tensors from participating client models.
    :param reference_tensor: Server tensor defining the expected shape, dtype, and device.
    :return: Aggregated tensor on the reference device and in its dtype.
    """
    if not isinstance(reference_tensor, Tensor):
        raise ValueError("FedBABU aggregation requires a tensor reference value.")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("FedBABU aggregation requires at least one client tensor.")

    for value in values:
        if not isinstance(value, Tensor):
            raise ValueError("FedBABU client state values must be tensors.")
        if value.shape != reference_tensor.shape:
            raise ValueError("FedBABU client tensor shape does not match the server model.")

    with torch.no_grad():
        if torch.is_floating_point(reference_tensor) or torch.is_complex(reference_tensor):
            aggregation_dtype = (torch.float32 if torch.is_floating_point(reference_tensor)
                                else reference_tensor.dtype)
            stacked_values = torch.stack([
                value.detach().to(device=reference_tensor.device, dtype=aggregation_dtype)
                for value in values
            ])
            return stacked_values.mean(dim=0).to(dtype=reference_tensor.dtype)

        stacked_values = torch.stack([
            value.detach().to(device=reference_tensor.device, dtype=reference_tensor.dtype)
            for value in values
        ])
        return torch.amax(stacked_values, dim=0)


def aggregate_fedbabu_body_parameters(model: nn.Module,
                                     client_model_parameters: Mapping) -> OrderedDict:
    """
    Equally aggregate client body state while excluding every classifier-head key.

    :param model: Server model defining the body schema and destination dtypes.
    :param client_model_parameters: Non-empty mapping of client IDs to model states.
    :return: Aggregated body state dictionary in server-model key order.
    """
    if not isinstance(client_model_parameters, Mapping) or not client_model_parameters:
        raise ValueError("FedBABU body aggregation requires at least one client upload.")

    server_body, _ = split_fedbabu_body_and_head(model)
    client_bodies = []
    for client_id, client_parameters in client_model_parameters.items():
        client_body, _ = split_fedbabu_body_and_head(model, client_parameters)
        if list(client_body) != list(server_body):
            raise ValueError("FedBABU body keys do not match the server model for client " +
                             str(client_id) + ".")
        client_bodies.append(client_body)

    aggregated_body = OrderedDict()
    for key, reference_tensor in server_body.items():
        aggregated_body[key] = aggregate_fedbabu_tensor_values(
            [client_body[key] for client_body in client_bodies], reference_tensor)

    return aggregated_body


def split_fedbabu_body_and_head(model: nn.Module,
                                model_parameters: Mapping = None) -> Tuple[OrderedDict, OrderedDict]:
    """
    Split a supported model state into its feature body and final linear head.

    :param model: Model exposing its model type through get_model_type().
    :param model_parameters: Optional state dictionary to split.
    :return: Ordered body and head state dictionaries.
    """
    if model is None or not callable(getattr(model, 'get_model_type', None)):
        raise ValueError("FedBABU requires a model with a get_model_type() method.")

    model_type = model.get_model_type()
    head_module_name = FEDBABU_HEAD_MODULE_BY_MODEL_TYPE.get(model_type)
    if head_module_name is None:
        raise ValueError("FedBABU does not support model type: " + str(model_type))

    modules = dict(model.named_modules())
    head_module = modules.get(head_module_name)
    if not isinstance(head_module, nn.Linear):
        raise ValueError("FedBABU expected the final classifier module " + head_module_name +
                         " to be a Linear layer for model type " + str(model_type) + ".")

    expected_state = model.state_dict()
    state = expected_state if model_parameters is None else model_parameters
    if not isinstance(state, Mapping):
        raise ValueError("FedBABU model state must be a mapping of parameter names to tensors.")
    if set(state) != set(expected_state):
        raise ValueError("FedBABU model state keys do not match the model architecture.")
    for key, expected_tensor in expected_state.items():
        tensor = state[key]
        if not hasattr(tensor, 'shape') or tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError("FedBABU model state tensor shape does not match the model for key " + key + ".")

    head_prefix = head_module_name + '.'
    expected_head_keys = {head_prefix + key for key in head_module.state_dict()}
    actual_head_keys = {key for key in state if key.startswith(head_prefix)}
    if actual_head_keys != expected_head_keys:
        raise ValueError("FedBABU classifier state keys do not match module " + head_module_name + ".")

    body_parameters = OrderedDict((key, value) for key, value in state.items()
                                  if not key.startswith(head_prefix))
    head_parameters = OrderedDict((key, value) for key, value in state.items()
                                  if key.startswith(head_prefix))
    if not body_parameters or not head_parameters:
        raise ValueError("FedBABU requires non-empty body and head state dictionaries.")

    return body_parameters, head_parameters
