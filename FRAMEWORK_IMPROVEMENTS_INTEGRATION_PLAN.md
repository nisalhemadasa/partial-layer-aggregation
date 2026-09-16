# Framework improvements integration plan

Status: Phases A and B implemented and CPU-validated; CUDA execution remains unverified. Phases C-G remain planned.

Scope: PGF plotting, client participation, diagnostic logging, model-distance diagnostics, evaluation, drift controls, and CPU fallback.

Reference: [FedTNT source at commit 692a0e15b869fcd8f164d88ce00687fe842784b1](https://github.com/Anonymide15/FedTNT/tree/692a0e15b869fcd8f164d88ce00687fe842784b1). Adapt selected behavior to this repository rather than copying whole modules. The reference contains incomplete configuration and removed strategy paths.

## 1. Scope and implementation constraints

- Preserve this repository's model layout, Adult dataset support, Dirichlet partitioning, and existing strategy implementations.
- Do not import FedTNT's optimizer, loss, ten-batch training cap, model architectures, or FedEx/FedTNT algorithm changes.
- Keep ordinary FedAvg equally weighted. Weighted server evaluation is a separate setting and must not change model aggregation weights.
- FedCollab is excluded. Ditto integration is a separate workstream; diagnostics should permit future personalized-model records without treating Ditto as implemented.
- Keep configuration in `main.py`, constants in `constants.py`, and reusable logic in the relevant module or `utils.py`.
- Preserve legacy performance-log formats and current experiment behavior through compatible defaults, except where CPU fallback and honoring an explicitly configured participation fraction are intentional changes.
- Analysis notebooks are outside these seven implementation phases. Produce documented logs that future notebooks can consume.

### Main-loop constraint

`AGENTS.md` currently states: "The main loop running in `federated_network/network.py` should not be modified."

Client selection is currently hard-coded to all clients inside that loop. Correct participation, diagnostic timing, and persistence need small explicit orchestration changes; they cannot be completed cleanly through strategy modules alone. Before those implementation steps, resolve the restriction with a narrowly scoped instruction permitting the required wiring. This plan does not itself override it.

Keep the aggregation/distribution/training order intact unless a separately reviewed participation design requires otherwise. Do not work around the restriction through monkey-patching, hidden side effects, or a duplicate simulation loop. PGF helpers, device handling, schemas, and standalone diagnostics can be developed independently.

## 2. Proposed configuration and compatibility defaults

Names below are proposed additions, not existing options. Validate them at startup and record their resolved values in experiment metadata.

| Setting | Proposed default | Meaning |
| --- | --- | --- |
| `plot_formats` | `('png', 'pdf', 'pgf')` | Attempt PGF alongside existing formats. |
| `pgf_strict` | `False` | Warn and continue if PGF export is unavailable. |
| `client_select_fraction` | Explicitly `1.0` in existing experiment configurations | Preserve full participation for baseline reproduction; fractions below one enable sampling. |
| `client_sampling_seed` | Experiment seed | Independent reproducible client-selection stream. |
| `diagnostic_logging_enabled` | `False` | Opt in to additional structured logs. |
| `model_distance_logging_enabled` | `False` | Opt in to potentially expensive model comparisons. |
| `model_distance_interval` | `1` | Collect every N aggregation events when enabled. |
| `client_evaluation_stage` | `local_after_training` | Alternative stages: `local_before_download`, `global_after_download`. |
| `server_metric_weighting` | `uniform` | Optional `train_samples` weighting of connected-client metrics. |
| `drifted_class_metrics_enabled` | `False` | Additional class-subset metrics where affected classes are defined. |
| `drifted_sample_proportion` | `1.0` | Fraction of each affected client's local training samples eligible for drift. |
| `is_adopt_drift` | `True` | Test data follows the drifted concept; false retains the original test concept. |
| `device` | `auto` | Prefer CUDA when available, otherwise CPU; allow explicit `cpu` or `cuda`. |

Map the existing `is_server_adaptability` option to its current evaluation stage. Reject conflicting legacy/new options rather than silently choosing one. Do not accidentally activate partial participation through the constructor's current `0.5` default: audit and make existing experiment choices explicit.

## 3. Phase A: PGF plotting

**Files:** `plot_utils/plotting.py`, `main.py`; `constants.py` if format names need shared constants.

### Implementation

1. Extend `configure_and_save_plot()` to save requested formats through one consistent path.
2. Create missing parent directories before saving any format.
3. Preserve existing PNG resolution, PDF output, labels, legends, and layout.
4. Enable `.pgf` export with a documented LaTeX requirement. Do not globally replace the plotting backend just to save PGF.
5. On a PGF failure, report the path and reason; retain successful PNG/PDF outputs. In strict mode, propagate the export failure. Do not hide PNG/PDF failures.
6. Ensure warnings distinguish attempted exports from successfully written artifacts.

### Acceptance checks

- One representative line plot and one distribution plot save to a new nested directory.
- With LaTeX available, compile a minimal document including the generated PGF and inspect text, legends, and sizing.
- Without LaTeX, PNG/PDF still save and PGF produces an actionable warning.
- Test strict failure behavior only for the PGF path; avoid adding tests for trivial formatting details.

## 4. Phase B: CPU fallback

**Files:** `models/utils.py`, `federated_network/client.py`, `federated_network/server.py`, `strategy/Oracle/oracle.py`, `strategy/FedRC/fedrc.py`, `main.py`; a small dependency-light device utility if needed to avoid circular imports.

### Implementation

1. Centralize device resolution. Resolve once per experiment and ensure models, data, auxiliary models, and strategy state use the same device.
2. For `auto`, select CUDA if available and otherwise CPU. If explicit `cuda` is unavailable, fail clearly rather than silently ignoring the request.
3. Audit active `.cuda()`, device-specific tensor creation, CUDA seeding, AMP/scaler use, and device assumptions in all existing strategy paths.
4. Guard CUDA-only operations and use a CPU-compatible training context without changing optimizer, loss, epochs, or batch limits.
5. Keep helper dependencies acyclic; device configuration must take effect before model construction and must not leave conflicting import-time `DEVICE` values.
6. Log requested device, resolved device, and PyTorch version. CPU fallback does not guarantee identical floating-point results across hardware.

### Acceptance checks

- A tiny CPU experiment completes model construction, local training, aggregation, evaluation, and log writing.
- Explicit CPU selection works even on a CUDA machine, when available for testing.
- Explicit unavailable CUDA fails with a clear message.
- Exercise auxiliary-model and multi-model state placement using small FedAU/FLUID and FedRC checks.
- Run a small CUDA regression if hardware is available; otherwise record it as unverified.

## 5. Phase C: Client participation

**Files:** `federated_network/network.py`, `federated_network/utils.py`, `federated_network/server.py`, `main.py`.

### Implementation

1. Validate `0 < client_select_fraction <= 1`; return all clients at one and otherwise sample `max(1, floor(fraction * client_count))` distinct clients. Reject an empty network.
2. Use a seeded sampling generator independent of data-loader and drift randomness.
3. Replace the all-client assignment with explicit sampling after resolving the main-loop constraint.
4. Resolve the existing aggregation-before-training lifecycle before wiring sampling: an upload must correspond to a recorded local update, not an arbitrary newly selected client's stale model. Associate each aggregation with the preceding local-training cohort, or document and review an alternative ordering. Treat initial training and the final aggregation explicitly.
5. Record separate training and aggregation participant IDs when their events occur at different round boundaries.
6. Replace `sampled_clients[client_id]` assumptions with client-ID maps. Aggregate the intersection of connected clients and actual upload participants for every strategy.
7. If a server has no contributors, retain its state. In a hierarchy, define and record which child servers contributed fresh updates; do not silently count unchanged branches as fresh uploads.
8. Unselected clients do not train or upload. Preserve their local model unless a documented strategy explicitly distributes to them. Keep data-drift progression independent of whether a client was sampled.
9. Preserve the full-participation warm-up initially, record it as a separate event, and document its cost. Do not mislabel it as partial-participation training.

### Acceptance checks

- Fractions one, one-half, and a tiny positive fraction select the expected counts without duplicates.
- Same seed and setup reproduce participant sequences; unrelated diagnostic settings do not change them.
- Non-contiguous and shuffled IDs work without indexing errors.
- An unselected client cannot influence an aggregate, even if its model is deliberately assigned a large outlier value.
- Verify no-upload servers, Oracle flat clusters, a small hierarchy, and existing FedRC/FedEx distribution paths.
- Trace two rounds plus warm-up/final aggregation to prove each uploaded model comes from the logged training event.

## 6. Phase D: Diagnostic logging

**Files:** `constants.py`, `log_utils/logging.py`, `federated_network/utils.py`, `federated_network/server.py`, `federated_network/network.py`.

### Implementation

1. Add `write_structured_log()` for nested pickle data, creating output directories safely. Leave legacy performance logs readable by existing consumers.
2. Define a versioned schema with explicit event phase, round index, client ID, server identity, and model role. Use server absolute IDs or `(depth, server_id)` where IDs repeat across levels.
3. Add the following files under the configured log directory:

| Log | Contents |
| --- | --- |
| `sampled_clients_log.pkl` | Training/upload cohorts, warm-up designation, and round/event association. |
| `client_sample_counts_log.pkl` | Local partition sizes; distinguish sampled/used training counts where available. |
| `server_state_log.pkl` | Server topology, active strategy, and client-parent assignments by event. |
| `aggregation_weights_log.pkl` | Actual contributors and coefficients applied at each aggregation. |
| `strategy_config_log.pkl` | Resolved strategy, sampling, evaluation, drift, diagnostics, seed, and device configuration. |

4. Capture aggregation coefficients where they are actually used. For current FedAvg, report equal contributor weights. Do not label sample-count ratios as applied weights when the algorithm does not use them.
5. For FedAU/FedEx and other composite methods, record component-specific operations/weights where needed rather than inventing one scalar weight per client. Mark unavailable fields explicitly.
6. Convert stored diagnostic tensors to detached CPU values. Avoid serializing entire clients, datasets, or models.
7. Persist at a defined run boundary; document whether interrupted runs produce diagnostics. Interval collection should control overhead without changing RNG state or training.

### Acceptance checks

- Round-trip nested records through pickle and validate required schema fields.
- Logged contributors and weights reconstruct a small known aggregate.
- Log an empty-contributor event without division by zero or fabricated weights.
- Legacy logs remain readable, and diagnostic logging on/off produces the same participant sequence and model updates.

## 7. Phase E: Model-distance diagnostics

**Files:** `distance_metrics/distance_metrics.py`, `federated_network/server.py`, `federated_network/utils.py`, `constants.py`, `federated_network/network.py`.

### Implementation

1. Add a helper comparing clients with their actual parent servers, supporting Oracle's flat layout and hierarchical server identities.
2. Fix the collection phase as after aggregation and before distribution, so diagnostics compare local models with the newly aggregated server model. Record that phase explicitly.
3. Save `model_distances_log.pkl` and `layer_distances_log.pkl`, keyed by event, server, client, and parameter/layer name.
4. Record participating status and the client's last local-update event, so stale unselected models are distinguishable from current contributors.
5. Use detached floating-point tensors for norms, validate matching keys/shapes, and define whether non-parameter buffers are included. Derive whole-model L2 as the square root of summed squared per-tensor distances, not the mean of layer norms.
6. For multi-model strategies such as FedRC, include model/cluster identity and compare corresponding models. Do not report an unused `client.model` as the strategy's model.
7. Make collection optional and interval-controlled; distinguish skipped collection from a measured zero distance.

### Acceptance checks

- Two clients with different parent servers are compared with the correct references.
- Hand-calculated tensor distances agree with whole-model and per-layer logs.
- Integer buffers cannot trigger norm dtype errors; key/shape mismatches are surfaced.
- Collection leaves parameters, gradients, RNG state, and subsequent training unchanged.

## 8. Phase F: Evaluation

**Files:** `models/utils.py`, `federated_network/client.py`, `federated_network/server.py`, `federated_network/utils.py`, `constants.py`, `main.py`; minimal simulation wiring as required.

### Implementation

1. Make the evaluated model and timing explicit: local model before download, server/global model after download but before training, or local model after training.
2. For global-model evaluation, evaluate the server model on client data without permanently overwriting persistent client state or altering strategy-specific download behavior.
3. Keep one entry per client in stable client-ID order, including unselected clients. Evaluate their retained local model for local stages, or their assigned server model for global stages. Record that they did not train.
4. Preserve the legacy local-model meaning of `client_log.pkl`. Store global-model metrics separately, for example `downloaded_global_client_log.pkl`, and record timing metadata for all modes.
5. Add optional drifted-class metrics. Mask both predictions and loss inputs to the selected classes. Report sample counts and an explicit missing value when no matching samples exist, not a fabricated zero accuracy.
6. Define affected classes from the current drift event/pattern, log the evaluated class set, and handle rotation-only cases explicitly rather than assuming a label-swap map always exists.
7. Add optional `train_samples` weighting to server metrics computed from connected-client evaluations. Keep `uniform` as the compatibility default. This changes metric averaging only.
8. Preserve direct server-test-set evaluation when configured, Oracle's per-server outputs, and FedRC's per-model outputs. Use documented handling for empty groups and missing metrics.
9. Record warm-up and round boundaries so client, server, class, and diagnostic logs can be aligned without relying on list offsets.

### Acceptance checks

- Known client metrics with train sizes one and three produce the expected uniform and weighted results.
- Distinct local/global model predictions prove the selected evaluation stage is correct and evaluation does not mutate the client model.
- Class-subset loss and accuracy both exclude unrelated classes; empty subsets are explicit.
- Partial participation preserves client identity and log dimensions for every evaluation stage.
- Oracle and FedRC metric structures remain usable by current consumers.

## 9. Phase G: Drift controls

**Files:** `data/utils.py`, `drift_concepts/drift.py`, `drift_concepts/utils.py`, `federated_network/client.py`, `main.py`; constructor wiring as needed.

### Implementation

1. Validate `drifted_sample_proportion` in `[0, 1]`. Select a seeded, persistent mask within each client's actual local partition.
2. Resolve nested `Subset` indices correctly. Do not sample from the entire backing dataset and present that as a fraction of the local partition.
3. Apply label swaps and supported rotations only to selected training samples. Zero means no training drift; one preserves full eligibility.
4. State mask semantics: initially, the fraction selects eligible samples from the entire local partition; label swapping acts only where eligible samples also belong to the specified classes. Log realized affected counts.
5. Isolate mutated data/labels between clients that share a backing dataset. Preserve an original reference where needed to prevent unintended cumulative rotations or swaps.
6. Implement `is_adopt_drift`: true evaluates against the new concept for affected classes/images; false keeps original test labels/images. Document that full affected-class test drift can differ from partial training drift.
7. Move label-swap, rotation, and auxiliary-data helpers into `drift_concepts/utils.py`. Preserve the existing FedAU/FLUID algorithm while making auxiliary-data targeting explicit for adoption versus rejection.
8. Handle synchronous and asynchronous client groups consistently, including nested IDs and deterministic selection. Masks follow clients across sampling rounds; log reassignment or reset policies for recurring drift.
9. Validate gradual/incremental schedules so fractional masks and scheduled drift intensity have a defined composition rather than accidentally applying the fraction twice.

### Acceptance checks

- Zero, partial, and full fractions change only eligible local samples with reproducible masks.
- Nested subsets and shared backing datasets do not cause cross-client mutation.
- Adoption changes the intended test concept; rejection preserves original test data.
- Repeated label swaps and rotations follow the specified schedule without accidental accumulation.
- Auxiliary-data creation does not mutate the main training dataset's labels.
- Cover synchronous/asynchronous groups and a client that remains unselected while drift progresses.

## 10. Implementation order and completion checklist

Recommended order: A (PGF) and B (CPU), then C (participation contract), D (logging), E (distances), F (evaluation), and G (drift controls). Define shared event/schema conventions before implementing C-F. Revalidate class-subset evaluation after G introduces adoption/rejection modes.

- [ ] Resolve the narrow main-loop wiring constraint before dependent implementation.
- [x] Implement and validate PGF export independently.
- [x] Implement and validate device selection without changing training behavior (CPU validated; CUDA hardware check unavailable).
- [ ] Implement participant selection and verify upload/training event association.
- [ ] Add truthful structured diagnostics and stable schema metadata.
- [ ] Add parent-aware model distances with explicit collection timing.
- [ ] Add evaluation stages, class metrics, and optional server-metric weighting.
- [ ] Add isolated partial drift and adoption/rejection behavior.
- [ ] Run one integrated small experiment: a few clients, a few rounds, partial participation, one drift transition, diagnostics enabled, and CPU execution.
- [ ] Run a full-participation compatibility check with new behavioral options disabled or at compatibility defaults.
- [ ] Check existing strategy paths with targeted small fixtures; clearly identify any pre-existing failure instead of removing that path.
- [ ] Validate CUDA and PGF/LaTeX where available; list unavailable checks explicitly.
- [ ] Update `README.md` and `AGENTS.md` to describe implemented behavior only, including the new CPU policy.

Use focused behavioral checks for simulation changes, not tests that merely repeat configuration assignments. Keep generated verification artifacts out of committed changes. Each completed phase should record files changed, checks performed, results, and remaining limitations; do not mark implementation complete based on this plan alone.

### Phase A completion record

- Updated `plot_utils/plotting.py`: PNG/PDF/PGF defaults, parent-directory creation, optional `plot_formats` and `pgf_strict`, actionable PGF warnings, and figure cleanup on export failures. Existing callers automatically gain PGF export; no simulation or `main.py` changes were needed.
- Documented usage and the LaTeX requirement in `README.md` and `AGENTS.md`.
- Validated the shared function in isolation from simulation imports using installed Matplotlib 3.4.3. Line and bar plots saved all three formats; XeLaTeX compiled both PGFs into a one-page document.
- Controlled failures verified optional PGF fallback, strict PGF error propagation, PNG error propagation, and figure cleanup. PNG/PDF remained available after PGF failures.
- MiKTeX required access outside the sandbox to its user cache/log directories for successful LaTeX validation. A full simulation was not run for this plotting-only change.

### Phase B completion record

- Added `device_utils.py` with lazy shared device resolution and explicit auto/cpu/cuda configuration. `main.py` configures before seeding/construction; selection prints requested/resolved device and PyTorch version. Persistent diagnostic metadata remains Phase D work.
- Updated client/server construction, model training/evaluation, and FedRC state placement to use the shared accessor. Removed unused Oracle CUDA initialization and replaced disabled AMP wrappers with equivalent full-precision operations. The simulation loop and training hyperparameters are unchanged.
- Added four focused regression checks in `tests/test_device.py`: selection/error behavior, exact training-update equivalence, client/server/auxiliary/RRT CPU execution, and FedRC model/optimizer/statistic consistency with a real fit step. All passed on Python 3.10 / PyTorch 2.8.0+cpu.
- Ran the real simulation constructor and unchanged loop for two synthetic MNIST-shaped clients and two rounds with Oracle/FedAvg aggregation; plotting and pickle logging completed. Only dataset loading was substituted, and PGF was disabled for this CPU smoke run. Drift was scheduled beyond the smoke-run window.
- CUDA auto-selection and forced-CPU precedence were checked with mocked availability; actual CUDA training was unavailable. Validation does not cover every dataset/model/strategy combination or active drift. The missing `ucimlrepo` dependency was installed only into an ignored workspace validation directory.
- Updated `README.md`, `AGENTS.md`, and `DEV_PROG.md` with implemented behavior and validation limits.
