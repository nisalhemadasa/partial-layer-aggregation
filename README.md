# FLUID-for_heterogeneous-clients

### Installation

Use Python 3.10 or newer and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and TorchVision using the command for your platform from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). This lets you choose the appropriate CPU or CUDA build. Then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

On Debian or Ubuntu, the following system packages are optional. Tk is only needed for interactive plotting; a LaTeX engine such as XeLaTeX is needed only to export PGF plots.

```bash
sudo apt-get install libx11-dev python3-tk tk-dev
```


### Running and device selection

Run the simulation with `python main.py`. The default path in `main()` requires CUDA and raises an error if CUDA is unavailable. The CIFAR-10 rotation comparison and rotation L-ablation branches explicitly use CPU; the other special branches use CUDA. To change the device, update the relevant `configure_device(...)` call in `main.py`.

The four top-level experiment switches in `main.py` are all disabled by default. With them unchanged, execution falls through to the legacy experiment block, which constructs multiple networks sequentially (beginning with Tiny ImageNet-200, 10 IID clients, 0 non-IID clients, and 20 training rounds). All `run_simulation(...)` calls in that block are currently commented out except the FedEx CIFAR-10 alpha sweep, which constructs a 10-client network for each alpha and runs 400 rounds. Commenting out a simulation call does not prevent its network constructor from loading data and creating clients/servers.

The switches enable the CIFAR-10 FedEx shared-layer ablation, the CIFAR-10 Scenario B rotation comparison, the CIFAR-10 rotation L-ablation, and the FedEx L=1 label-swap matrix. The matrix currently lists Fashion-MNIST and Scenario B. Each early-dispatch switch returns after its experiment.

Datasets are loaded from `datasets/`. MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 use TorchVision download support when data is missing; Adult data is fetched through `ucimlrepo`, and Tiny ImageNet uses the repository's custom loader. These downloads require network access on first use.

For programmatic use, call `device_utils.configure_device('auto'|'cpu'|'cuda')` once before constructing clients, servers, or optimizers. Auto mode selects CUDA when available and otherwise CPU; explicitly requesting unavailable CUDA raises an error. Modules use `get_device()` for placement. Changing the setting does not migrate existing models or optimizer state. The requested device, resolved device, and PyTorch version are printed at selection.

Training uses the configured optimizer, loss, scheduler, and epoch/batch settings. CPU and GPU runs may produce different numerical results.

Focused regression checks run with `python -m unittest discover -s tests -p test_device.py` in a Python environment with the project's dependencies (including `ucimlrepo`, imported by the dataset loader).

### Plot export

The shared `configure_and_save_plot()` function saves PNG (300 DPI), PDF, and PGF files and creates missing output directories. Existing plotting calls enable all three formats automatically.

PGF export requires a working LaTeX installation with the engine selected by Matplotlib's `pgf.texsystem` available on `PATH` (normally `xelatex`). Include a successful export in a LaTeX document using `\usepackage{pgf}` and `\input{path/to/plot.pgf}`.

If PGF export fails, the function warns with the output path and cause while retaining PNG/PDF output. A failed export may leave an incomplete PGF file; use it only after a successful export. PNG/PDF errors still propagate. Direct callers can pass `plot_formats=('png', 'pdf')` to skip PGF or `pgf_strict=True` to raise on PGF failures. Figures are closed after export, including when saving fails.

### FedEx shared-layer ablation notation

In this code, `L` is the number of model layers shared and aggregated by FedEx. The paper uses `L` for the number of layers kept local, which is `n - L` in this code, where `n` is the model's total number of layers. For example, the CIFAR-10 CNN here has four named layers (`conv1`, `conv2`, `fc1`, and `fc2`), so code `L=1` means one shared layer and three local layers; this corresponds to `L=3` in the paper.

### Model-distance diagnostics

Each aggregation records Euclidean (L2) distances after server aggregation and before model distribution. Whole-model L2 is computed as the square root of the summed squared differences across all state tensors; per-tensor distances are retained separately. Integer buffers are safely converted for comparison.

The current run's canonical structure is available as `fed_net.model_distance_history`. Its records are keyed by round, phase, leaf server, model identity, and client. Each client record contains the whole-model distance, layer distances, participation status, and last local-update round. `client_distances()` provides a compact primary-model mapping intended for future FedEx optimization.

Set `model_distance_logging_enabled=False` in `simulation_parameters` to disable collection, or set `model_distance_interval=N` to collect every N aggregation rounds. Defaults are enabled and every round. Completed runs write `model_distances_log.pkl` and `layer_distances_log.pkl` under `log_save_path`. FedRC records each corresponding cluster model separately.

### Evaluation

`client_log.pkl` continues to contain post-training metrics for each client's local model. Configure an additional structured evaluation stage with `client_evaluation_stage` in `simulation_parameters`:

- `local_before_download`: evaluate local client models before server aggregation/distribution.
- `global_after_download`: evaluate each client's assigned leaf-server model after distribution and before local training, without overwriting the client model.
- `local_after_training`: evaluate local client models after local training; this is the default.

Completed runs write the selected stage to `evaluation_log.pkl`. Global/downloaded-model runs also write `downloaded_global_client_log.pkl`. These records include client and assigned-server identity, model role, participation status, loss, and accuracy.

Set `drifted_class_metrics_enabled=True` to include label-swap class-subset loss, accuracy, and sample counts in `drifted_class_log.pkl`. Rotation-only drift produces an empty class set and zero matching samples. `server_metric_weighting` controls client-derived server metrics: use `uniform` (default) or `train_samples`. This affects reported server metrics only, not aggregation.

### Implemented strategies

`constants.RecoveryAlgorithm` includes FedAvg, FedAU, FedEx, FedRC, Oracle, FLUID, RRT, Ditto, FairFedDrift, and FedBABU. The strategy packages currently include FedAvg, FedAU, FedEx, FedRC, Oracle, Ditto, FairFedDrift, and FedBABU; FLUID and RRT use the framework's existing client/server paths. Support and constraints vary by method, so check the strategy configuration and validation in `federated_network/server.py` and `federated_network/network.py` before combining methods or server layouts.

### Ditto integration

Ditto currently supports fixed-lambda personalized training with full client participation and a single server level. Each client keeps `client.model` for upload/aggregation and a persistent `client.ditto_personal_model` for personalization. Ditto global aggregation is weighted by local sample count; existing FedAvg aggregation remains uniform.

`client_log.pkl` continues to describe `client.model`. When `ditto_eval_personalized=True`, personalized metrics are evaluated after local training and written separately to `ditto_personalized_client_log.pkl`. With `drifted_class_metrics_enabled=True`, personalized label-drift subset metrics are written to `ditto_personalized_drifted_class_log.pkl`. `ditto_state_log.pkl` records the resolved configuration and model-role metadata.

Set `ditto_dynamic_lambda=True` to reserve a persistent seeded validation partition for each client and select among `ditto_lambda_candidates` after comparable candidate updates. Training never uses the held-out validation indices. Per-round, per-client candidate losses and selected values are written to `ditto_selected_lambda_log.pkl`.
