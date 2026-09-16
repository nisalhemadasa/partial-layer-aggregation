# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This repository is a PyTorch simulation framework for federated learning under concept drift, identified in `README.md` as `FLUID-for_heterogeneous-clients`. The main entry point is `main.py`, which configures datasets, drift behavior, federated network topology, recovery methods, and experiment output paths.

Core source folders:

- `data/`: dataset loading, preprocessing, and IID/non-IID partitioning.
- `drift_concepts/`: concept-drift definitions and drift application logic.
- `federated_network/`: client, server, topology, simulation loop, aggregation, and distribution flow.
- `models/`: PyTorch model definitions and train/test routines.
- `strategy/`: federated aggregation and recovery strategies.
- `distance_metrics/`: model-distance calculations.
- `log_utils/`: metric aggregation and log persistence.
- `plot_utils/`: plot generation.

## Files and Folders to Avoid

Do not inspect or modify these unless explicitly requested:

- `venv/`
- `.git/`
- `.idea/`
- generated plot/log output files

## Development Notes

- After every development step, append a very brief dated entry to `DEV_PROG.md` describing the change, validation result, and any remaining limitation. Include code and documentation changes; distinguish completed work from plans and unverified claims.
- Prefer small, scoped edits that match the existing research-code style.
- Keep experiment configuration changes in `main.py` unless a reusable behavior belongs in a module.
- The main loop running in `federated_network/network.py` should not be modified.
- Use the existing strategy pattern under `strategy/` when adding or changing aggregation methods.
- The third party repositories (aggregation methods) that are being integrated into the project should be added as a strategy similar to the existing ones. 
Their instances should be created in a similar way to the existing ones (e.g., Oracle, FedAvg, etc.), which then will interact with the main loop in `federated_network/network.py`.
- Use existing constants from `constants.py` instead of adding duplicate string literals.
- Helper functions should be added in a cleaner way, i.e., use `utils.py` whenever possible, and keep the main logic in the core modules.
- Keep dataset handling inside `data/` and drift-specific behavior inside `drift_concepts/`.
- Model-distance diagnostics use `ModelDistanceHistory` and are captured after aggregation and before distribution. Preserve this timing and structure because future FedEx optimization will consume it. Whole-model distance must remain true Euclidean L2 over all included tensors; do not replace it with a sum of layer norms.
- Preserve `client_log.pkl` as post-training local-model metrics. Staged evaluation uses `evaluation_log.pkl`; `global_after_download` evaluates assigned server models without mutating `client.model`. Drifted-class metrics apply only to label-swap drift and must record the matching test-sample count. `server_metric_weighting` changes reported client-derived server metrics only, never aggregation weights.
- Use `device_utils.get_device()` for device placement. Configure `auto`, `cpu`, or `cuda` through `configure_device()` before constructing models and optimizers; do not introduce fixed import-time CUDA devices. Auto mode falls back to CPU, while explicit unavailable CUDA raises an error. Do not change devices during an existing experiment.
- Use the existing multiline docstring style for functions and methods: start with a short description, then list `:param ...:` entries and `:return:` when applicable.
- Keep the existing vertical spacing style between functions: two blank lines between top-level functions and one blank line between class methods.

## Existing Strategies and Integration Scope

- Existing strategy packages are `FedAvg`, `FedAU`, `FedEx`, `FedRC`, and `Oracle` under `strategy/`.
- `constants.RecoveryAlgorithm` also includes `FLUID` and `RRT`; their behavior uses existing client/server paths rather than dedicated strategy packages.
- Strategy creation and switching live in `federated_network/server.py` (`server_fn` and `change_server_aggregation_strategy`). Client training lives in `federated_network/client.py`, with orchestration helpers in `federated_network/utils.py` and model training routines in `models/utils.py`.
- Existing FedAvg uses an equal average of model tensors, not sample-count weighting. Do not assume it accepts client sample counts or silently change aggregation for existing methods.
- Oracle uses a flat multi-server layout; preserve evaluation of all its cluster servers.
- FedCollab is not implemented and will not be integrated. Do not add FedCollab strategies, configuration, or dependencies.
- Ditto is planned but not yet implemented. The requirements below describe the intended integration, not existing functionality.
- `integration_progress.txt` contains inherited claims about FedCollab and weighted FedAvg that do not describe this checkout. Treat it as historical context. Verify references in `Ditto_integration_plan.txt` against the current code; that plan also contains inherited assumptions.

## Planned Ditto Integration

- Implement Ditto under `strategy/Ditto/` and add `constants.RecoveryAlgorithm.DITTO` for selection.
- Keep the integration PyTorch-native; do not import the upstream TensorFlow Ditto implementation.
- Keep Ditto-specific logic in `strategy/Ditto/` where possible, with client state and small wiring changes in the existing client/server modules and orchestration helpers. Preserve the main simulation loop.
- Keep `client.model` as the local/upload model used for aggregation, and introduce `client.ditto_personal_model` as the separate personalized model.
- Implement sample-count weighted FedAvg for Ditto over `client.model`; do not upload or aggregate `client.ditto_personal_model`. Existing FedAvg does not provide sample weighting, so add an explicit Ditto path or an opt-in weighted helper that preserves existing methods' behavior.
- Keep `client_log.pkl` consistent with the rest of the framework: it records metrics for `client.model`.
- Ditto personalized metrics belong in Ditto-specific logs such as `ditto_personalized_client_log.pkl` and `ditto_personalized_drifted_class_log.pkl`.
- If dynamic lambda selection is implemented, `ditto_dynamic_lambda=True` should reserve a validation split from each client's sampled local training data, choose among `ditto_lambda_candidates`, and log selected values in `ditto_selected_lambda_log.pkl`. These options are not currently available.
- Robust aggregation options from upstream Ditto are out of scope for the first integration; add them later as independent aggregation strategies if needed.

## Running

Typical run command:

```bash
python main.py
```

Before running, inspect all active `FederatedNetwork(...)` constructors and `fed_net.run_simulation(...)` calls in `main.py`. Multiple experiments can execute sequentially, including parameter sweeps. Commenting out a simulation call does not disable its network constructor, which loads data and creates clients and servers. For validation, ensure only the intended small experiment is active.

## Testing and Verification

Focused device/training regression checks are in `tests/test_device.py`; run `python -m unittest discover -s tests -p test_device.py` with project dependencies installed. For changes to simulation behavior, also use the smallest feasible experiment configuration in `main.py` and run `python main.py` on the selected device. A comprehensive simulation test suite is not yet available.

Focused model-distance checks are in `tests/test_model_distance_diagnostics.py`.

Focused staged-evaluation checks are in `tests/test_evaluation.py`.

For documentation-only changes, read the edited Markdown file back after writing it.

## Output Locations

Simulation output paths are controlled by arguments passed to `FederatedNetwork.run_simulation(...)`.

Common outputs:

- plots under `plots/...`
- logs under `logs/...`

The shared `plot_utils/plotting.py::configure_and_save_plot()` saves PNG, PDF, and PGF by default and creates missing output directories. PGF requires a working LaTeX engine; failures warn while preserving PNG/PDF. Direct callers can select `plot_formats` or enable `pgf_strict=True` to raise on PGF errors.

Avoid committing generated experiment artifacts unless the user specifically asks for them.
