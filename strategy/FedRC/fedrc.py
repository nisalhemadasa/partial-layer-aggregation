"""
Description: This file consists supporting functions for FedRC (Robust Clustering) algorithm.

Y. Guo, X. Tang, and T. Lin, “FedRC: Tackling Diverse Distribution Shifts Challenge in Federated Learning by Robust
Clustering,” in Proceedings of the 41st International Conference on Machine Learning (ICML 2024), 2024.

Assumption: No adding or removing (refer to the appendix H.4.) of global models (θ_k) are done.

Author: Nisal Hemadasa
Date: 04-12-2025
Version: 1.0
"""
from typing import List, OrderedDict, Dict

import torch
from torch import nn
import torch.nn.functional as F

import constants
from federated_network.client import Client

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class FedRC:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def aggregate_models(self, fedrc_server_models: List[nn.Module],
                         client_model_params_dict: Dict[str, List[OrderedDict]]) -> None:
        """
        Aggregate the client models to the global model and returns the new aggregated model.
        *** Here we use FedAvg, as the orginal paper allows using any aggregation method ***
        # TODO: This function is not designed to the hierarchical server topology
        :param fedrc_server_models: The list of models (cluster bases) in the server model
        :param client_model_params_dict: Dictionary containing the client IDs (keys) and the corresponding state dicts
        of all the models (fedrc_cluster_count amount of models) adopted by each client
        return: None
        """
        for idx in range(len(fedrc_server_models)):
            server_model_params = fedrc_server_models[idx].state_dict()

            if client_model_params_dict is not None:
                for client in client_model_params_dict.keys():
                    assert len(client_model_params_dict[client]) == len(fedrc_server_models), \
                        "Number of FedRC models in the client and server must be the same."
                    fedrc_client_models = client_model_params_dict[client]
                client_model_params_list = client_model_params_dict.values()

            # Simple averaging of weights
            updated_server_model_params = server_model_params.copy()
            for key in updated_server_model_params.keys():
                updated_server_model_params[key] = torch.stack(
                    [client_model_params[key].float() for client_model_params in client_model_params_list], 0).mean(0)

            # Load the FedAvg-ed parameters to the server model
            # set_parameters(server_model, updated_server_model_params)


def get_fedrc_client_model_params(client: Client) -> List[OrderedDict]:
    """
    Return a list of state_dicts for all FedRC models of a client.
    :param client: The client whose FedRC model parameters are to be retrieved
    :return: List of state_dicts for all FedRC models of the client
    """
    return [model.state_dict() for model in client.fedrc_models]


def fit(client: Client) -> None:
    """
    Train the client model using training set, and using gamma. Then update the ω and C values for the client.

    I.e., this function implements Eq.(3) and Eq.(5) for a single client, using the approximations from the paper:

        - f(x, y; θ_k) = cross-entropy loss ---- (A)
        - P(y|x; θ_k) ≈ exp(-f) ---------------- (B)
        - P(y; θ_k) ≈ C_{y,k} ------------------ (C)
        - I_tilde(x, y; θ_k) ≈ exp(-f) / C_{y,k} ----- (D)

        Label-wise accumulation for C_{y,k} => I.e., we calculate the proportion of the data pairs labeled as y,and that
        chooses the cluster/model k.
        C_{y,k} = (1/N) · ∑_i ∑_j 1{ y_{i,j} = y } · γ_{i,j;k} / (1/N) · ∑_i ∑_j γ_{i,j;k} --------(E)

        Eq. (3) is renamed into two parts for clarity:
        γᵢⱼ,ₖᵗ = ( ωᵢ,ₖ^(t−1) · Ĩ(xᵢⱼ, yᵢⱼ; θₖ^(t−1)) ) / ∑ₙ=1...K [ ωᵢ,ₙ^(t−1) · Ĩ(xᵢⱼ, yᵢⱼ; θₙ^(t−1)) ]  ------ Eq. (3.1)
        ω_{i,k}^(t) = (1 / N_i) ∑_j γ_{i,j;k} ------ Eq. (3.2)

    Algorithm in a nutshell => This is an EM (Expectation–Maximization) algorithm with the following steps:
    ┌──────────────────┐
    │   OLD (ω, C, θ)  │
    └───────┬──────────┘
            │
            ▼
    ┌──────────────────┐
    │   E-step         │
    │ compute γ        │
    └───────┬──────────┘
            │
            ▼
    ┌──────────────────┐
    │   M-step         │
    │ update (ω, C, θ) │
    └───────┬──────────┘
            │
            ▼
    repeat until convergence

    :param client: client to perform the FedRC training step, and update its ω and C values
    :return: None
    """
    fedrc_models = client.fedrc_models  # List[nn.Module], length K
    num_clusters = len(fedrc_models)
    fedrc_optimizers = client.fedrc_optimizers  # Optimizers for weights train of the fedrc_models

    prev_omega = client.omega_i_k.to(DEVICE)  # [K]
    prev_C = client.C_y_k.to(DEVICE)  # [num_classes, K]

    # Initiate variables to hold the values of numerator and denominator of Eq. (E)
    sum_gamma_per_cluster = torch.zeros(num_clusters, device=DEVICE)  # ∑_j γ_{i,j;k} --- from denominator of (E)
    label_gamma = torch.zeros(client.num_classes, num_clusters, device=DEVICE)  # ∑_j 1{y_j=y} γ_{i,j;k} ---- from numerator of (E). Shape: [num_classes, K]
    num_samples = 0

    # Loop over local data
    # In the paper, one sample at a time is used, but here we use minibatches for efficiency (faster on GPU), but
    # is mathematically equivalent (just vectorized).
    for x, y in client.trainloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)  # B = batch size -> Label vector y is a 1D tensor of length [B]. This is equal to the 'minibatch_size' given in network.py.
        batch_size = y.size(0)  # y.size(0) == B
        num_samples += batch_size

        # Compute per-sample loss for each cluster model
        # losses_all: [B, K]
        losses_all = torch.empty(batch_size, num_clusters, device=DEVICE)

        for k, model_k in enumerate(fedrc_models):
            model_k.to(DEVICE)
            model_k.eval()  # we only evaluate here

            with torch.no_grad():
                logits = model_k(x)  # get the logits in a tenson of dimension [B, num_classes]. Represented by m_{i,k}(x, θ_k) in appendix H.2. in the paper.
                losses_all[:, k] = F.cross_entropy(logits, y, reduction="none")  # per-sample cross-entropy loss, from ----(A), and appendix H.2. in the paper.
                # losses_all shape: [B, K]

        # ===================================
        # constructing Eq. (D)
        # I_tilde - this I_tilde value is then used to compute Eq. (3.1)
        # ===================================
        # exp(-loss) term: [B, K]
        # B -> mini_batch_size
        # K -> number of fedrc_models (taken from the for-loop above) which is equivalent to number of clusters.
        exp_neg_loss = torch.exp(-losses_all)  # from (B) ---- (B.numerator)

        # C_{y,k} term for this batch of samples.
        # prev_C[y, k] shape: [B, K]
        C_batch = prev_C[y]  # gather rows by label y, from(C) ---- (B.denominator)
        # To avoid C_batch (C_{y,k}) being zero (NaN), we clamp it to a minimum value equivalent to EPS, i.e.,
        # if C_batch < EPS, then C_batch = EPS
        # For EPS, a very small value (1e-12), which is achievable in data type floating point precisions, is chosen
        # to avoid numerical instability.
        EPS = 1e-12
        C_batch = C_batch.clamp_min(EPS)  # avoid division by 0 and getting NaN | ---- (B.denominator)

        # I_tilde(x, y; θ_k) = exp(-f) / C_{y,k}, from ----- (D)
        I_tilde = exp_neg_loss / C_batch  # I_tilde shape: [B, K] | ---- (B.numerator / B.denominator)

        # =============================================================================
        # constructing Eq. (3.1), by applying I_tilde from (D) in (3.1)
        # New γ : Sample weights for client i, sample j, and cluster K
        # =============================================================================
        # Numerator for γ = ω_{i;k}^{t-1} * I_tilde
        # prev_omega: [K] -> [1, K]
        numerators = prev_omega.unsqueeze(0) * I_tilde  # [B, K] | ----- (3.1.numerator)

        # Denominator: sum over clusters n, from Eq. (3.1) (clamped to EPS to avoid division by 0 and getting NaN)
        denominator = numerators.sum(dim=1, keepdim=True).clamp_min(EPS)  # [B, 1] | ----- (3.1.denominator)

        # γ_{i,j;k}^{(t)}: [B, K], from Eq. (3.1)
        gamma_batch = numerators / denominator  # ------- (3.1.numerator / 3.1.denominator) = Eq. (3.1)

        # ===================================================================
        # Constructing Eq. (3.2) - part 1
        # ===================================================================
        # ∑_j γ_{i,j;k}^{(t)} (from Eq. (3.1)), (i.e., Accumulates γ_{i,j;k}^{(t)} over j)
        # sums over the batch dimension B, so: input: [B, K], output: [K]
        sum_gamma_per_cluster += gamma_batch.sum(dim=0)  # [K] | from Eq. (3.2) ----- Eq. (3.2.1)

        # ===============================================================================================
        # Constructing (E) - part 1, using the calculated new γ_{i,j;k} values from Eq. (3.1)
        # ===============================================================================================
        # Numerator for New C_{y,k} = ∑_j 1{y_j=y} · γ_{i,j;k},
        # where 1{y_j=y} is an indicator function. i.e., 1{y_j=y} = 1, if y_j == y, else 0
        # Label-wise accumulation of gamma values
        # label_gamma: [num_classes, K]
        label_gamma.index_add_(0, y, gamma_batch)  # ------ (E.numerator)

        # ========================================================================
        # constructing Eq. (5)
        # Update each theta_k -> calculate gradients and perform an optimizer step
        # ========================================================================
        # Now we need gradients, so we re-forward with grad on
        for k, model_k in enumerate(fedrc_models):
            model_k.to(DEVICE)
            model_k.train()
            optimizer_k = fedrc_optimizers[k]
            optimizer_k.zero_grad()

            logits_k = model_k(x)   # [B, num_classes]
            per_sample_loss_k = F.cross_entropy(logits_k, y, reduction="none")  # [B]

            # Eq. (5): weighted loss over this batch for cluster k
            # L_k ≈ mean_j γ_{i,j;k} * Cross_Entropy_loss_on_cluster_k(x_j, y_j)
            gamma_ijk  = gamma_batch[:, k].detach()           # [B], no grad through gamma, gamma_ijk -> per sample cluster weights
            loss_k = (gamma_ijk  * per_sample_loss_k).mean()

            # Backward pass and optimization
            loss_k.backward()
            optimizer_k.step()

    if num_samples == 0:
        # No data, keep previous omega/C unchanged
        return

    # ===================================================================
    # Constructing Eq. (3.2) - part 2
    # new ω : ω_{i,k}^(t) : Cluster weights for client i and cluster K
    # ===================================================================
    # ω_{i,k}^(t) = (1 / N_i) ∑_j γ_{i,j;k}
    new_omega = sum_gamma_per_cluster / float(num_samples)  # ----- Eq. (3.2.1) * (1 / N_i) = Eq. (3.2) #TODO:change naming to new_omega_i_k?

    # ===============================================================================================
    # Constructing (E) - part 2
    # ===============================================================================================
    # Denominator for New C_{y,k} = ∑_j γ_{i,j;k} -> (new γ from Eq. (3.1))
    # To Eq. (3.2.1), add numerical-safety: avoid division by zero
    cluster_weight_total = sum_gamma_per_cluster.clamp_min(EPS)  # [K] | ----- (E.denominator)  #TODO:change naming gamma_ijk_total?

    # ===============================================================================================
    # Constructing (E) - part 3
    # new C_{y,k} : How much of cluster k is made up of label y
    # ===============================================================================================
    # C_{y,k}^(t) = (∑_j 1{y_j=y} γ_{i,j;k}) / (∑_j γ_{i,j;k})
    # label_gamma: [num_classes, K]
    new_C_y_k = label_gamma / cluster_weight_total.unsqueeze(0)  # ----- (E.numerator) / (E.denominator) = Eq. (E)

    # =========================================================================================================
    # Numerical safety, when, denominator of C_{y,k} is extremely low (cluster k has almost no samples assigned)
    # (This part is not specified in the paper)
    # ==========================================================================================================
    # For numerical safety: handle clusters with extremely low samples assigned.
    # low_mass_mask (extremely lower γ value) is a boolean vector of shape [K].If cluster_weight_total less than
    # 1e-6, we declare that cluster
    # (e.g.,k=3) to have low mass (i.e. low_mass_mask[3] = [True]).
    low_mass_mask = (cluster_weight_total < 1e-6)
    # Then, if cluster k has almost no samples assigned (γ <<< ), it's C_y,k is set to 1.0.
    # Why 1.0? This way C_{y,k} will
    #   1. have a neutral, non-informative prior, equally likely across all labels,
    #   2. prevent NaNs and keeps the algorithm numerically stable.
    if low_mass_mask.any():
        # For all labels y, where γ <<<, set C_{y,k} = 1.0
        new_C_y_k[:, low_mass_mask] = 1.0

    # Save back to client (detach stops tracking it in PyTorch’s computation graph (treats it like a NumPy array))
    client.omega_i_k = new_omega.detach()
    client.C_y_k = new_C_y_k.detach()


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedRC(strategy_name=constants.RecoveryAlgorithm.FEDRC)
    return _strategy
