# FLUID-for_heterogeneous-clients

### Dependencies
pip install \
scipy==1.14.1
matplotlib==3.9.2
torch==2.9.1
torchvision=0.20.0

apt-get install -y \
libx11-dev
python3-tk tk-dev

### Device selection

`main.py` calls `configure_device('auto')` before seeding and constructing the experiment. Auto mode uses CUDA when available and otherwise CPU. Change this argument to `'cpu'` to force CPU, or `'cuda'` to require CUDA (an unavailable CUDA device raises a clear error).

Programmatic callers should call `device_utils.configure_device(...)` before creating clients, servers, or optimizers. Otherwise, the first device access resolves auto mode. Modules use `get_device()` rather than importing a fixed device value. Configure once before construction; changing the setting does not migrate an existing experiment or optimizer state. The requested device, resolved device, and PyTorch version are printed at selection.

Training remains full precision with the existing optimizer, loss, scheduler, and epoch/batch settings. CPU execution does not imply numerical equivalence across hardware.

Focused regression checks run with `python -m unittest discover -s tests -p test_device.py` in a Python environment with the project's dependencies (including `ucimlrepo`, imported by the dataset loader).

### Plot export

The shared `configure_and_save_plot()` function saves PNG (300 DPI), PDF, and PGF files and creates missing output directories. Existing plotting calls enable all three formats automatically.

PGF export requires a working LaTeX installation with the engine selected by Matplotlib's `pgf.texsystem` available on `PATH` (normally `xelatex`). Include a successful export in a LaTeX document using `\usepackage{pgf}` and `\input{path/to/plot.pgf}`.

If PGF export fails, the function warns with the output path and cause while retaining PNG/PDF output. A failed export may leave an incomplete PGF file; use it only after a successful export. PNG/PDF errors still propagate. Direct callers can pass `plot_formats=('png', 'pdf')` to skip PGF or `pgf_strict=True` to raise on PGF failures. Figures are closed after export, including when saving fails.

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
