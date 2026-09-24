# FedBABU integration plan

Status: Read-only study completed on 2026-09-24; implementation has not started.

## Goal

Add FedBABU as a selectable strategy using this repository's PyTorch models,
client/server lifecycle, logging, and experiment configuration. Keep the
iterative simulation loop in `federated_network/network.py` unchanged. Implement
only FedBABU-specific behavior around the current strategy pattern and reuse
existing framework helpers where their behavior matches the algorithm.

FedBABU separates a model into a body (feature extractor) and a head
(classifier). A single initial head is shared by all clients and stays fixed
throughout federated training. Clients update their bodies, the server averages
and redistributes those bodies, and the unchanged head remains attached to each
complete model. After federated training, each client's head is fine-tuned on
that client's local training data for personalized evaluation. The upstream
paper defines the head as the final linear layer. See the [paper](https://openreview.net/pdf?id=HuaYQfggn5u)
and [official implementation](https://github.com/jhoon-oh/FedBABU).

## Important difference from FedEx

FedEx is the closest structural precedent: `strategy/FedEx/fedex.py` aggregates
only selected feature-extractor parameters, while
`federated_network/server.py::model_distribution_fedex()` sends the server
extractor back to clients. But current `Client.fit()` trains all parameters for
FedEx, leaving each client's classifier local. FedBABU must instead keep one
common classifier unchanged during all federated rounds, train only the body,
and fine-tune a client-specific head only after federated training.

Do not change FedEx's behavior or assume its existing
`models/utils.py::split_to_extractor_and_classifier()` is a FedBABU-compatible
split. That helper currently classifies several `fc*` layers as classifier
parameters for many model types; FedBABU's head is the final classifier layer.
Add a separate explicit body/head mapping for FedBABU, and preserve the existing
FedEx helper and behavior.

## Adaptation boundaries

- Keep the feature selectable through `constants.RecoveryAlgorithm`,
  `server_fn()`, and the ordinary experiment handles in `main.py`.
- Keep aggregation behavior inside `strategy/FedBABU/`; keep client-specific
  training and state changes in `federated_network/client.py` and narrowly
  scoped helpers. Reuse the existing server strategy dispatch and normal model
  distribution when they preserve the fixed shared head.
- Keep the framework's optimizer, loss, batch processing, device selection,
  client metric logs, staged evaluation, and model-distance timing intact unless
  a FedBABU-specific helper is needed. Never silently change FedAvg or FedEx.
- Preserve `client.model` as the model used by the existing training and upload
  path. Fine-tuned client heads are personalized evaluation models and must not
  be uploaded or mixed into `client_log.pkl`.
- Treat FedBABU as the selected strategy for the complete experiment; do not
  switch to it only after a drift event. The fixed-head federated phase spans
  initialization/warm-up and all communication rounds. Its head fine-tuning is
  a post-training personalization/evaluation phase, not a drift detector or
  drift-triggered recovery action.
- Add no separate runner, model framework, or upstream dependency. Do not edit
  the repeated-round loop in `federated_network/network.py`. If a required
  behavior cannot be connected without changing that loop, document the exact
  need and find an outside-loop integration point first.
- Support model architectures only after verifying their final classifier
  layer and state-dict keys. Do not claim support for an architecture whose
  body/head split has not been checked.

## Stepwise implementation

Complete one small item at a time. After each development step, append a brief
dated entry to `DEV_PROG.md` with the change, validation result, and remaining
limitation. Do not mark a planned test as passed before it runs.

### 0. Verify model and lifecycle assumptions

[ ] 0.1 Inspect model definitions under `models/` and list each supported
    architecture's final classifier module and body/head state-dict keys.
    Confirm whether every server and client begins with the same head, including
    the framework's round-0/warm-up path and any configured model initialization.

[ ] 0.2 Trace `FederatedNetwork.run_simulation()`,
    `federated_network/utils.py::train_client_models()`,
    `federated_network/server.py::model_aggregation()` and the existing
    distribution path. Record where FedBABU training and post-run personalized
    evaluation can be called without editing the repeated-round loop.

    Check: document the actual initialization, upload, aggregation,
    distribution, evaluation and logging order before implementation. Confirm
    the full-participation behavior stated in `AGENTS.md` and identify any
    constraints on this first integration.

### 1. Register the strategy

Files: `constants.py`, `strategy/__init__.py`,
`strategy/FedBABU/__init__.py`, `strategy/FedBABU/fedbabu.py`,
`federated_network/server.py`.

[ ] 1.1 Add `constants.RecoveryAlgorithm.FEDBABU` without changing existing
    strategy constants.
[ ] 1.2 Add a small strategy class and `aggregator_fn()` following
    `strategy/FedEx/fedex.py` and `strategy/Ditto/ditto.py` conventions.
[ ] 1.3 Register construction in `server.py::server_fn()` and aggregation
    dispatch in the existing `Server.train()` strategy path.
[ ] 1.4 Add focused checks in `tests/test_fedbabu_strategy.py` for constant,
    package import, factory selection, and server strategy selection.

    Check: the strategy can be selected without altering FedAvg, FedEx, or the
    simulation loop. Aggregation may explicitly remain unimplemented until
    step 3.

### 2. Define FedBABU's exact body and head

Files: `strategy/FedBABU/utils.py` (or a narrowly scoped helper in
`models/utils.py` only if it is genuinely reusable),
`tests/test_fedbabu_strategy.py`.

[ ] 2.1 Implement a FedBABU-specific model/state-dict split that identifies the
    final classifier layer for each architecture accepted in step 0. Do not
    edit the existing FedEx split helper.
[ ] 2.2 Add checks that body and head keys are disjoint, cover the full model
    state, preserve tensor shapes, and identify exactly the final classifier.
[ ] 2.3 Decide handling for buffers (including integer counters) explicitly;
    retain the framework's valid state-dict dtypes when averaging/updating.
[ ] 2.4 Reject unsupported model types with a clear message instead of guessing
    layer names.

    Check: changing the new helper has no effect on FedEx's existing split or
    aggregation behavior.

### 3. Implement body-only server aggregation

Files: `strategy/FedBABU/fedbabu.py`, `strategy/FedBABU/utils.py`,
`federated_network/server.py` only if dispatch requires it,
`tests/test_fedbabu_strategy.py`.

[ ] 3.1 Aggregate only client body parameters using the current FedAvg
    aggregation convention unless the upstream FedBABU setting for the tested
    configuration requires otherwise. Do not add sample weighting by
    modifying existing FedAvg.
[ ] 3.2 Apply the aggregated body to the server model while leaving its shared
    head exactly unchanged.
[ ] 3.3 Confirm how the framework distributes the complete server model.
    Reuse `model_distribution_hierarchy()` if it broadcasts the same fixed head
    and updated body correctly; add a FedBABU-specific path only if required.
[ ] 3.4 Test that body values aggregate as expected, the server head is
    bit-identical before/after aggregation, and distribution gives clients the
    aggregated body plus the same unchanged head.

    Check: empty/mismatched contributor sets fail clearly; existing FedAvg and
    FedEx aggregation tests remain unchanged.

### 4. Train only the body on clients

Files: `federated_network/client.py`, optionally
`models/utils.py` for a parameter-selective training helper,
`federated_network/utils.py` only if the existing orchestration needs a narrow
strategy branch, `tests/test_fedbabu_strategy.py`.

[ ] 4.1 Route FedBABU client training through an explicit method before the
    ordinary full-model `train()` path. Cover round-0/warm-up as well as normal
    rounds so the classifier is never accidentally trained during the
    federated phase.
[ ] 4.2 Reuse the existing local dataset, loss convention, optimizer family,
    epoch count, batch traversal, and `device_utils.get_device()` behavior.
    Restrict optimizer updates to body parameters; do not copy upstream
    optimizer/data-loop code wholesale.
[ ] 4.3 Ensure all client heads are initialized from the same shared head and
    remain unchanged across client training, aggregation, and downloads.
[ ] 4.4 Follow the participation behavior confirmed in step 0 while preserving
    the framework's current simulation semantics; do not introduce sampling or
    edit the main loop.
[ ] 4.5 Test parameter deltas: body changes after training, head does not;
    repeat across warm-up, stationary rounds, and after a configured drift.

    Check: ordinary client training and all non-FedBABU strategies behave
    exactly as before.

### 5. Add post-training personalized head fine-tuning

Files: `strategy/FedBABU/fedbabu.py` or `strategy/FedBABU/utils.py`,
`federated_network/client.py`, and an outside-loop call site in
`main.py` or an existing post-run orchestration method, selected after step 0.
Reuse structured evaluation/logging helpers in `federated_network/utils.py`
and `log_utils/` where suitable.

[ ] 5.1 After the final federated round, create a per-client evaluation model
    from the final shared body and the common fixed head.
[ ] 5.2 Fine-tune only that model's classifier on the client's local training
    data. Keep the trained body fixed. Make fine-tuning epochs and optimizer
    settings explicit, validated FedBABU configuration rather than hidden
    constants.
[ ] 5.3 Evaluate the personalized model on the existing client test/validation
    path without using test data for fine-tuning. Keep these records separate
    from `client_log.pkl` and identify their model role explicitly.
[ ] 5.4 Reuse existing output/log formats where possible; do not introduce
    repository-side PaLA evaluation functions or change general evaluation
    semantics.
[ ] 5.5 Ensure this finalization is invoked after the selected FedBABU
    `run_simulation()` call without editing the iterative loop. If current
    persistence boundaries make that impossible, stop and document the minimal
    conflict before changing `network.py`.

    Check: a test proves personalized evaluation changes only the client head,
    leaves the shared/server model unchanged, and does not replace the
    existing post-training client metrics.

### 6. Add configuration and experiment handles

Files: `main.py`, `federated_network/network.py` only for constructor/config
validation (not the simulation loop), `tests/test_fedbabu_strategy.py`.

[ ] 6.1 Define FedBABU parameters next to the current recovery configuration:
    classifier fine-tuning epochs and any needed classifier learning rate.
    Keep defaults explicit and validate positive finite values.
[ ] 6.2 Create disabled-by-default FedBABU experiment handles in the same
    dataset sections and with the same construction/run structure as existing
    FedAvg/FedEx/Oracle handles. Use the framework's existing drift scenarios,
    output naming, and device setup.
[ ] 6.3 Limit active handles during checks as required by `AGENTS.md`; a
    commented-out simulation call does not disable its constructor.

    Check: every planned handle constructs the FedBABU strategy and routes
    through the ordinary simulation entry point.

### 7. Validate incrementally

[ ] 7.1 Run focused strategy tests: partition, fixed-head behavior, body-only
    aggregation/distribution, parameter-selective local training, and
    post-training personalized evaluation.
[ ] 7.2 Run existing strategy, device, staged-evaluation, and model-distance
    tests that are available with installed dependencies.
[ ] 7.3 Run the smallest feasible full-participation CPU simulation for one
    supported image model. Check ordinary logs, distinct personalized outputs,
    unchanged server/client fixed heads during training, and drift continuity.
[ ] 7.4 Run one matching FedAvg/FedEx compatibility smoke and verify their
    outputs/behavior were not changed.
[ ] 7.5 Record unavailable dependencies, hardware checks, and unsupported model
    types as limitations; do not claim GPU or benchmark validation unless run.

## Out of scope for the first integration

- Changing the main simulation loop, FedAvg, FedEx, or existing model
  definitions solely to match upstream naming.
- FedBABU variants that train the head on server data, alternative body/head
  update modes, and unrelated robust aggregation methods.
- Changing dataset labels, drift specifications, or the framework's metric
  calculation functions.
- Treating the end-of-training personalized head evaluation as an online
  per-round adaptation method; that would be a separate algorithmic extension.
