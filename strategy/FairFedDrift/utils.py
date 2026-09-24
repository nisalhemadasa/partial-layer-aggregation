"""Current decision data and local history for the PaLA adaptation of FairFedDrift."""
import math
import hashlib
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import DatasetSnapshot, convert_dataset_to_loader
from device_utils import get_device


def average_fairfeddrift_parameters(parameter_sets, sample_counts):
    """
    Build sample-weighted parameters using Ditto's existing weighting convention.
    Integer/bool buffers come from the largest contributor (first on ties).
    :param parameter_sets: Nonempty ordered list of matching model state dictionaries.
    :param sample_counts: Positive integer counts in the same order as parameter_sets.
    :return: Independent averaged state dictionary; input tensors remain unchanged.
    """
    if not parameter_sets or len(parameter_sets) != len(sample_counts):
        raise ValueError("FairFedDrift averaging requires matching models and sample counts.")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in sample_counts):
        raise ValueError("FairFedDrift sample counts must be positive integers.")
    reference = parameter_sets[0]
    for parameters in parameter_sets:
        if list(parameters) != list(reference):
            raise ValueError("FairFedDrift model parameter keys must match.")
        for key, tensor in reference.items():
            other = parameters[key]
            if other.shape != tensor.shape or other.dtype != tensor.dtype:
                raise ValueError("FairFedDrift tensor shapes and dtypes must match: " + key)
    largest = max(range(len(sample_counts)), key=sample_counts.__getitem__)
    total = sum(sample_counts)
    averaged = OrderedDict()
    with torch.no_grad():
        for key, tensor in reference.items():
            if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                value = torch.zeros_like(tensor)
                for parameters, count in zip(parameter_sets, sample_counts):
                    value.add_(parameters[key].detach().to(tensor.device), alpha=count / total)
                averaged[key] = value
            else:
                averaged[key] = parameter_sets[largest][key].detach().to(tensor.device).clone()
    return averaged


def calculate_fairfeddrift_merge_distances(cluster_models, histories, assignments,
                                          timestep_start_round: int, loss_threshold: float,
                                          batch_size: int = 64):
    """
    Compare models on retained historical data grouped by cluster and client.
    Scalar adaptation of teresalazar13/FairFedDrift's fed_drift.py merge rule,
    revision 51dec5eb1dc46a0c77e0ac0aa59c5724a46fab9a (CC BY 4.0):
    https://github.com/teresalazar13/FairFedDrift/blob/51dec5eb1dc46a0c77e0ac0aa59c5724a46fab9a/federated/algorithms/drift/fed_drift.py
    Infinity replaces the upstream 1000 sentinel for ineligible pairs.
    :param cluster_models: Active cluster ID to model mapping.
    :param histories: Client ID to ClientDataHistory, already advanced to the current round.
    :param assignments: Client ID to arrival-round/cluster-ID mapping for retained data.
    :param timestep_start_round: First round of the current timestep; exclude it and later arrivals.
    :param loss_threshold: Shared finite nonnegative scalar threshold, equality passes.
    :param batch_size: Positive evaluation batch size; partial batches are included.
    Clusters without retained history keep infinite distances and are not evaluated.
    :return: Symmetric nested distance mapping and historical sample counts per active cluster.
    """
    if (isinstance(timestep_start_round, bool) or not isinstance(timestep_start_round, int)
            or timestep_start_round < 0):
        raise ValueError("Timestep start must be a non-negative round.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Merge evaluation batch size must be a positive integer.")
    if (isinstance(loss_threshold, bool) or not isinstance(loss_threshold, (int, float))
            or not math.isfinite(loss_threshold) or loss_threshold < 0):
        raise ValueError("Merge loss threshold must be finite and non-negative.")
    grouped = {cluster_id: {} for cluster_id in cluster_models}
    for client_id, history in histories.items():
        if history.current_round < timestep_start_round:
            raise ValueError("Advance history to the current round before merge evaluation.")
        for arrival_round, snapshot in history.records():
            if arrival_round >= timestep_start_round:
                continue
            if arrival_round not in assignments.get(client_id, {}):
                raise ValueError("Retained merge data has no historical cluster assignment.")
            cluster_id = assignments[client_id][arrival_round]
            if cluster_id not in cluster_models:
                raise ValueError("Historical merge data references an inactive cluster.")
            grouped[cluster_id].setdefault(client_id, []).append(snapshot)
    counts = {cluster_id: 0 for cluster_id in cluster_models}
    distances = {i: {j: math.inf for j in cluster_models} for i in cluster_models}
    # Losses only for model/data clusters with retained history.
    losses = {model_id: {} for model_id in cluster_models}
    history_cluster_ids = [cluster_id for cluster_id, clients in grouped.items() if clients]
    for cluster_id in history_cluster_ids:
        counts[cluster_id] = sum(len(snapshot) for snapshots in grouped[cluster_id].values()
                                 for snapshot in snapshots)
    if len(history_cluster_ids) < 2:
        return distances, counts
    for cluster_id in history_cluster_ids:
        clients = grouped[cluster_id]
        for snapshots in clients.values():
            dataset = ConcatDataset(snapshots)
            loader = convert_dataset_to_loader(dataset, batch_size, _is_shuffle=False)
            for model_id in history_cluster_ids:
                model = cluster_models[model_id]
                losses[model_id].setdefault(cluster_id, []).append(
                    evaluate_fairfeddrift_loss(model, loader))
    ids = history_cluster_ids
    for position, i in enumerate(ids):
        for j in ids[position + 1:]:
            if not counts[i] or not counts[j]:
                continue
            # max over every cross-client pair equals max(cross) - min(own).
            distance = max(0.0, max(losses[i][j]) - min(losses[i][i]),
                           max(losses[j][i]) - min(losses[j][j]))
            if distance <= loss_threshold:
                distances[i][j] = distances[j][i] = distance
    return distances, counts


def update_fairfeddrift_complete_linkage(distances, first_id: int, second_id: int,
                                       merged_id: int):
    """
    Produce distances after a merge using the maximum constituent distance.
    :param distances: Symmetric nested mapping from the historical merge calculation.
    :param first_id: First active cluster to merge.
    :param second_id: Distinct second active cluster to merge.
    :param merged_id: New cluster ID, absent from the input mapping.
    :return: New mapping excluding retired IDs, without modifying the input.
    """
    if (first_id == second_id or first_id not in distances or second_id not in distances
            or merged_id in distances):
        raise ValueError("Complete linkage requires two active IDs and a new merged ID.")
    remaining = [i for i in distances if i not in (first_id, second_id)]
    updated = {i: {j: distances[i][j] for j in remaining} for i in remaining}
    updated[merged_id] = {merged_id: math.inf}
    for i in remaining:
        value = max(distances[first_id][i], distances[second_id][i])
        updated[i][merged_id] = updated[merged_id][i] = value
    return updated


def evaluate_fairfeddrift_loss(model: nn.Module, decision_loader: DataLoader) -> float:
    """
    Evaluate one candidate's sample-mean cross-entropy on current decision data.
    Reuse the same prepared loader for all candidates. The model must already be
    on the configured device; evaluation does not move it or update gradients.
    :param model: Candidate cluster model on the configured experiment device.
    :param decision_loader: Frozen current training samples prepared in step 2.5.
    :return: Finite mean loss over all evaluated samples, without sensitive groups.
    """
    criterion = nn.CrossEntropyLoss(reduction='sum')
    modes = [(module, module.training) for module in model.modules()]
    loss_sum, sample_count = 0.0, 0
    device = get_device()
    try:
        model.eval()
        with torch.no_grad():
            for inputs, labels in decision_loader:
                labels = labels.to(device)
                outputs = model(inputs.to(device))
                loss_sum += criterion(outputs, labels).item()
                sample_count += labels.size(0)
    finally:
        # Restore mixed submodule modes as well as the root mode, even on failure.
        for module, training in modes:
            module.training = training
    if sample_count == 0:
        raise ValueError("FairFedDrift candidate loss requires nonempty decision data.")
    loss = loss_sum / sample_count
    if not math.isfinite(loss):
        raise ValueError("FairFedDrift candidate loss must be finite.")
    return loss


def prepare_fairfeddrift_decision_loader(client) -> DataLoader:
    """
    Capture current training samples once for all candidate model comparisons.
    Call after drift application and Client.sample_data(), before local training.
    Reuse the returned loader for every candidate; rebuilding it would resample
    random input transformations. The existing client sampler selects a Subset
    stored in trainloader.dataset; every sample in that subset is captured.
    This helper neither reads evaluation data nor appends to historical data.
    :param client: Client with a current ordinary training DataLoader.
    :return: Non-shuffled loader over an independent snapshot of current samples.
    """
    trainloader = getattr(client, 'trainloader', None)
    if not isinstance(trainloader, DataLoader):
        raise ValueError("FairFedDrift decision data requires a current client.trainloader.")
    batch_size = trainloader.batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("FairFedDrift decision data requires a positive training batch size.")
    if len(trainloader.dataset) == 0:
        raise ValueError("FairFedDrift decision data cannot be empty.")
    snapshot = DatasetSnapshot(trainloader.dataset)
    return convert_dataset_to_loader(snapshot, batch_size, _is_shuffle=False)


class ClientDataHistory:
    """One client's arrivals; cluster assignments are managed by the strategy separately."""

    def __init__(self, window: int | None = 100):
        """
        Initialize bounded history without retaining any live dataset references.
        :param window: Positive retention length in communication rounds, or None.
        """
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("History window must be None or a positive integer of rounds.")
        self.window = window
        self.current_round = -1
        self._last_arrival_round = -1
        self._snapshots = {}
        self._arrival_fingerprints = {}
        self._snapshot_cache = {}
        self._snapshot_reference_counts = {}

    def advance(self, round_idx: int) -> list[int]:
        """
        Advance the communication clock and release expired snapshots.
        :param round_idx: Nondecreasing round index; warm-up is 0, first training round is 1.
        :return: Expired arrival rounds for later assignment-history cleanup.
        """
        if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
            raise ValueError("History round must be a non-negative integer.")
        if round_idx < self.current_round:
            raise ValueError("History cannot move backwards in communication rounds.")
        self.current_round = round_idx
        cutoff = 0 if self.window is None else max(0, round_idx - self.window + 1)
        expired = [arrival for arrival in self._snapshots if arrival < cutoff]
        for arrival in expired:
            del self._snapshots[arrival]
            fingerprint = self._arrival_fingerprints.pop(arrival)
            self._snapshot_reference_counts[fingerprint] -= 1
            if self._snapshot_reference_counts[fingerprint] == 0:
                del self._snapshot_reference_counts[fingerprint]
                del self._snapshot_cache[fingerprint]
        return expired

    @staticmethod
    def _fingerprint(snapshot: DatasetSnapshot) -> bytes:
        """Return a stable digest of the captured input/label version."""
        digest = hashlib.sha256()
        for inputs, label in snapshot._samples:
            for tensor in (inputs, label):
                contiguous = tensor.detach().cpu().contiguous()
                digest.update(str(contiguous.dtype).encode('ascii'))
                digest.update(repr(tuple(contiguous.shape)).encode('ascii'))
                digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.digest()

    def add(self, arrival_round: int, dataset) -> bool:
        """
        Capture one arrival without refreshing or duplicating an existing arrival.
        :param arrival_round: Current round of a new batch, or retained arrival round for reuse.
        :param dataset: Selected local dataset to snapshot only when this is a new arrival.
        :return: True for a new snapshot, False for repeated use of an arrival.
        """
        if isinstance(arrival_round, bool) or not isinstance(arrival_round, int) or arrival_round < 0:
            raise ValueError("History arrival round must be a non-negative integer.")
        if arrival_round in self._snapshots or arrival_round == self._last_arrival_round:
            return False
        if arrival_round < self.current_round or arrival_round < self._last_arrival_round:
            raise ValueError("Cannot insert an old arrival into current history.")
        # Validate/materialize first so a failed capture leaves the clock/history unchanged.
        captured_snapshot = DatasetSnapshot(dataset)
        fingerprint = self._fingerprint(captured_snapshot)
        snapshot = self._snapshot_cache.get(fingerprint, captured_snapshot)
        self.advance(arrival_round)
        self._snapshots[arrival_round] = snapshot
        self._arrival_fingerprints[arrival_round] = fingerprint
        self._snapshot_cache[fingerprint] = snapshot
        self._snapshot_reference_counts[fingerprint] = self._snapshot_reference_counts.get(fingerprint, 0) + 1
        self._last_arrival_round = arrival_round
        return True

    def records(self) -> list[tuple[int, DatasetSnapshot]]:
        """
        Return retained arrivals in order, without exposing the history mapping.
        :return: Arrival-round and snapshot pairs; consumers must release expired references.
        """
        return list(self._snapshots.items())


def build_fairfeddrift_history_loaders(history: ClientDataHistory, assignment_by_arrival: dict,
                                       batch_size: int, before_round: int | None = None) -> tuple[dict, dict]:
    """
    Build one retained-data loader and sample count for each cluster assigned to this client.
    Identical captured versions assigned repeatedly to one cluster are used once, while
    the same version can independently contribute to different clusters.
    :param history: Client's bounded, detached arrival history.
    :param assignment_by_arrival: Arrival-round to learned cluster ID mapping.
    :param batch_size: Positive local training batch size.
    :param before_round: Optional exclusive round cutoff for the current timestep's new arrival.
    :return: Cluster ID to shuffled DataLoader and cluster ID to unique sample count mappings.
    """
    if not isinstance(history, ClientDataHistory):
        raise ValueError("FairFedDrift loaders require a ClientDataHistory instance.")
    if not isinstance(assignment_by_arrival, dict):
        raise ValueError("FairFedDrift historical assignments must be a dictionary.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("FairFedDrift history loader batch size must be a positive integer.")
    if before_round is not None and (isinstance(before_round, bool) or
                                     not isinstance(before_round, int) or before_round < 0):
        raise ValueError("FairFedDrift history cutoff must be a non-negative round.")

    grouped = {}
    for arrival_round, snapshot in history.records():
        if before_round is not None and arrival_round >= before_round:
            continue
        if arrival_round not in assignment_by_arrival:
            raise ValueError("Retained FairFedDrift data is missing its historical cluster assignment.")
        cluster_id = assignment_by_arrival[arrival_round]
        if isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 0:
            raise ValueError("Historical FairFedDrift cluster IDs must be non-negative integers.")
        # Snapshot identity is stable because ClientDataHistory interns exact data versions.
        grouped.setdefault(cluster_id, {}).setdefault(id(snapshot), snapshot)

    loaders, sample_counts = {}, {}
    for cluster_id, unique_snapshots in grouped.items():
        dataset = ConcatDataset(list(unique_snapshots.values()))
        loaders[cluster_id] = convert_dataset_to_loader(dataset, batch_size, _is_shuffle=True)
        sample_counts[cluster_id] = len(dataset)
    return loaders, sample_counts
