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
- Restrict the first integration to a single-level server layout. The existing
  `model_distribution_hierarchy()` updates child servers by calling their
  aggregation strategy with both child and parent models, which would average
  bodies rather than copy one canonical FedBABU body downward. Supporting
  hierarchies needs a separate design and is out of scope for this first pass.
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

[x] 0.1 Inspect model definitions under `models/` and list each supported
    architecture's final classifier module and body/head state-dict keys.
    Confirm whether every server and client begins with the same head, including
    the framework's round-0/warm-up path and any configured model initialization.

    Audit result: active dataset/model pairs use `CNNModel.fc2` (MNIST and
    Fashion-MNIST), `CNNCIFAR10.fc2`, `ResNet18CIFAR100.fc2`,
    `ConvNeXtTinyImageNet.head`, and `TabularAdultModel.fc2`. The constructors
    in `client_fn()` and `server_fn()` independently initialize fresh models;
    therefore clients and servers do not start with one common classifier head.
    `run_simulation()` calls `client_initial_training()` before the first
    communication round, and that helper calls `Client.fit()` with no server
    parameters. The first implementation must establish one common head across
    clients and all servers before this warm-up, then keep it fixed throughout
    warm-up and later federated rounds. A narrow constructor-time initialization
    hook is the current candidate; confirm it in step 0.2. No model definitions
    need to change for these explicit final-layer mappings.

[x] 0.2 Trace `FederatedNetwork.run_simulation()`,
    `federated_network/utils.py::train_client_models()`,
    `federated_network/server.py::model_aggregation()` and the existing
    distribution path. Record where FedBABU training and post-run personalized
    evaluation can be called without editing the repeated-round loop.

    Audit result: `run_simulation()` calls `client_initial_training()` before
    entering its round loop, so the shared-head initialization and body-only
    `Client.fit()` route must already be active there. During rounds,
    `train_client_models()` loads the assigned full server state into sampled
    clients before calling `Client.fit()`; this existing download also restores
    the common fixed head. Add FedBABU to `model_aggregation()`'s ordinary
    client-to-server strategy branch and to `Server.train()` dispatch so only
    the body is aggregated. Flat topology has no intermediate server
    distribution. Because the existing hierarchical path calls `Server.train()`
    with parent and child state dictionaries, require a single-level layout for
    the first integration rather than misrepresent that operation as an exact
    broadcast. After the repeated-round loop and before its existing log writes,
    `run_simulation()` has an outside-loop point to build per-client personalized
    evaluation models, fine-tune their heads, and persist separate results. No
    edit to the loop body or separate `main.py` post-run call is needed.

    Decision: the first integration will support a flat, single-server layout
    and use an outside-loop finalization block before existing log persistence.
    Hierarchical FedBABU remains unsupported until its distribution semantics
    are separately designed.

    Check: document the actual initialization, upload, aggregation,
    distribution, evaluation and logging order before implementation. Audit
    the current loop's client participation directly; the full-participation
    restriction in `AGENTS.md` is specific to Ditto and should not be treated
    as a new FedBABU requirement.

### 1. Register the strategy

Files: `constants.py`, `strategy/__init__.py`,
`strategy/FedBABU/__init__.py`, `strategy/FedBABU/fedbabu.py`,
`federated_network/server.py`.

[x] 1.1 Add `constants.RecoveryAlgorithm.FEDBABU` without changing existing
    strategy constants. Added value `fedbabu`; imported and asserted its value
    successfully. No strategy behavior is wired yet.
[x] 1.2 Add a small strategy class and `aggregator_fn()` following
    `strategy/FedEx/fedex.py` and `strategy/Ditto/ditto.py` conventions.
    Created `strategy/FedBABU/` and exported its strategy module from
    `strategy/__init__.py`. The factory returns the FedBABU recovery name;
    `aggregate_models()` raises `NotImplementedError` until body aggregation is
    implemented in step 3.
[x] 1.3 Register construction in `server.py::server_fn()` and aggregation
    dispatch in the existing `Server.train()` strategy path. Added the strategy
    factory case, routed FedBABU leaf servers through the existing ordinary
    client-to-server aggregation helper, and added a dedicated `Server.train()`
    dispatch to its aggregation method. Runtime test confirmed all three routes
    reach the selected FedBABU strategy; its method still explicitly raises
    `NotImplementedError` until step 3.
[x] 1.4 Added focused checks in `tests/test_fedbabu_strategy.py` for the
    recovery constant, package import, factory, server strategy selection, and
    explicit placeholder dispatch. All four tests passed with Python 3.10.

    Check: the strategy can be selected without altering FedAvg, FedEx, or the
    simulation loop. Aggregation may explicitly remain unimplemented until
    step 3.

### 2. Define FedBABU's exact body and head

Files: `strategy/FedBABU/utils.py` (or a narrowly scoped helper in
`models/utils.py` only if it is genuinely reusable),
`tests/test_fedbabu_strategy.py`.

[x] 2.1 Implemented `strategy/FedBABU/utils.py::split_fedbabu_body_and_head()`
    with explicit final-head mappings for the five active model types identified
    in step 0.1. It verifies that the configured module is linear and that the
    supplied state contains its expected keys. Exported the helper from the
    FedBABU package; left the existing FedEx split helper unchanged. Seven
    focused FedBABU tests passed, including mapping selection and rejection of
    unknown/non-linear architectures.
[x] 2.2 Added checks that body and head keys are disjoint, cover the full model
    state, preserve tensor shapes, and identify exactly the final classifier.
    The helper also rejects incomplete/unexpected state keys and shape mismatch
    against the model schema. Supplied states are split without mutation. All
    nine focused FedBABU tests passed with Python 3.10.
[x] 2.3 Added `aggregate_fedbabu_tensor_values()` as the explicit tensor
    reduction primitive for the later body aggregator. Floating and complex
    state uses equal averaging; non-floating buffers use elementwise maximum,
    preserving the server dtype (so BatchNorm counters remain integral). The
    active CIFAR-100 ResNet audit found 20 integer `num_batches_tracked`
    buffers; its running means/variances are floating-point buffers and use the
    mean. Added tests for float means, integer counters, boolean buffers, dtype
    preservation, invalid/empty inputs, and shape mismatch. All 13 focused
    FedBABU tests passed with Python 3.10. The strategy aggregation method will
    consume this helper in step 3.1.
[x] 2.4 Reject unsupported model types with a clear message instead of guessing
    layer names. Implemented with the step 2.1 helper and covered by the
    unsupported-model test; no separate code change was needed here.

    Check: changing the new helper has no effect on FedEx's existing split or
    aggregation behavior.

### 3. Implement body-only server aggregation

Files: `strategy/FedBABU/fedbabu.py`, `strategy/FedBABU/utils.py`,
`federated_network/server.py` only if dispatch requires it,
`tests/test_fedbabu_strategy.py`.

[x] 3.1 Added `aggregate_fedbabu_body_parameters()` to validate client uploads,
    split out only body state, and equally reduce each body tensor using the
    step 2.3 policy. The result excludes the head and follows server body key
    order. No client sample weighting was introduced and existing FedAvg stays
    unchanged. Tests cover equal means, large opposing head values, arbitrary
    client IDs, no input/server mutation, empty uploads and invalid model-state
    schemas. All 15 focused FedBABU tests passed with Python 3.10. Applying the
    returned body to the server remains step 3.2.
[x] 3.2 Updated `FedBABU.aggregate_models()` to apply the result from step 3.1
    through the existing `models/utils.py::set_parameters()` helper with
    non-strict loading, so the server body changes and all omitted head tensors
    stay unchanged. A strategy-level test verified averaged body values and
    bit-identical server-head tensors. All 16 focused FedBABU tests passed with
    Python 3.10.
[x] 3.3 Confirmed the existing distribution path for the supported flat layout.
    `model_distribution_hierarchy()` has no intermediate servers to update in
    this topology. In `train_client_models()`, FedBABU follows the generic
    branch that loads the complete assigned `server.model.state_dict()` into
    each participating client before local training. This sends both the new
    body and unchanged fixed head; no FedBABU-specific distribution function is
    required. A focused runtime test verified all body and head tensors match
    the server after the ordinary download path.
[x] 3.4 Tested that body values aggregate as expected, the server head is
    bit-identical before/after aggregation, and distribution gives clients the
    aggregated body plus the same unchanged head. Added normal-path
    aggregation/download checks, and verified empty, missing-key, and
    wrong-shape uploads raise before mutating server state. Added compatibility
    checks showing FedAvg still averages its complete model state and FedEx
    still updates its existing extractor partition only. All 21 focused
    FedBABU tests passed. The Ditto regression suite could not import because
    this environment lacks `ucimlrepo` (required during network-module import).

    Check: empty/mismatched contributor sets fail clearly; existing FedAvg and
    FedEx aggregation tests remain unchanged. No production code changed in
    this validation step.

### 4. Train only the body on clients

Files: `federated_network/client.py`, optionally
`models/utils.py` for a parameter-selective training helper,
`federated_network/utils.py` only if the existing orchestration needs a narrow
strategy branch, `tests/test_fedbabu_strategy.py`.

[x] 4.1 Routed FedBABU client training through `Client._fit_fedbabu()` before
    the ordinary full-model `train()` path. It calls
    `strategy/FedBABU/utils.py::train_fedbabu_body()`, which temporarily freezes
    final-head parameters and reuses `models.utils.train()` for the existing
    optimizer, loss, scheduler and batch traversal. The route applies to
    round-0/warm-up and every later round, regardless of drift state. Focused
    checks confirm body updates, unchanged local heads, and restored original
    gradient flags; all 23 FedBABU tests passed. Client/server heads are still
    independently initialized at this point; common initialization is step 4.3.
[x] 4.2 Confirmed `train_fedbabu_body()` passes the existing client
    `DataLoader` and configured client epoch count to `models.utils.train()`.
    The existing trainer continues to own its `NLLLoss`, SGD optimizer and
    scheduler, batch traversal, and `get_device()` placement; FedBABU adds no
    duplicate training loop or optimizer settings. A focused delegation test
    checks the exact loader/epoch handoff and that only body parameters require
    gradients and enter the optimizer during the call. All 24 FedBABU tests
    and all four device tests passed on CPU; only pre-existing AMP deprecation
    warnings were emitted.
[x] 4.3 Added `initialize_fedbabu_shared_head()` and call it from
    `FederatedNetwork.__init__()` after clients and servers are constructed and
    before `run_simulation()` performs local warm-up. The first flat server
    supplies the canonical initial head; only head state is copied, and model
    head schemas are checked before any writes. A focused test confirms all
    clients receive that head while all server/client bodies stay unchanged.
    The 26-test FedBABU suite and source compilation passed. The constructor
    integration is compile-checked; a live network-constructor run remains
    unavailable here because `ucimlrepo` is not installed. Later client
    training and aggregation tests continue to verify head invariance.
[x] 4.4 Confirmed the current `run_simulation()` round loop assigns
    `sampled_clients = self.clients` and passes all clients to training and
    aggregation. Although `sample_clients()` and `client_select_fraction`
    exist, the loop does not call them. FedBABU therefore follows the current
    full-participation behavior without adding sampling or changing the loop.
    The Ditto-specific full-participation constraint in `AGENTS.md` is not an
    independent FedBABU constraint. The existing flat-layout requirement still
    applies to this first FedBABU integration.
[x] 4.5 Verified parameter deltas across initial local warm-up, a regular
    server-backed round, and the drift-active `Client.fit()` path: the body
    changes while the client head remains fixed. Added a regression check that
    FedAvg still enters the existing unrestricted full-model trainer. The
    27-test FedBABU suite passed. These are focused phase/path checks using the
    drift-active flag; a full simulation driven by a configured drift scenario
    remains in step 7.3.

    Check: ordinary client training and all non-FedBABU strategies behave
    exactly as before. FedAvg/FedEx aggregation compatibility is also covered
    by step 3.4; broader strategy and configured-drift simulation checks remain
    in step 7.

### 5. Add post-training personalized head fine-tuning

Files: `strategy/FedBABU/fedbabu.py` or `strategy/FedBABU/utils.py`,
`federated_network/client.py`, and an outside-loop finalization block in
`federated_network/network.py` before its existing log persistence.
Reuse structured evaluation/logging helpers in `federated_network/utils.py`
and `log_utils/` where suitable.

[x] 5.1 Added `initialize_fedbabu_personal_models()` to create one independent
    evaluation-model copy per client from the final flat server model. This
    gives each copy the final shared body and common fixed head, validates
    client/server architecture schemas, and leaves client/server models
    untouched. Focused tests verify identical starting state and independent
    copies; all 28 FedBABU tests passed. Step 5.5 now calls this helper after
    the repeated-round loop.
[x] 5.2 Added `Client.fine_tune_fedbabu_personal_head()` and
    `train_fedbabu_head()`. Fine-tuning uses the full local training set, not
    the framework's 10% ordinary training subset; only classifier parameters
    enter SGD. Body parameters are frozen, body modules stay in evaluation mode
    so BatchNorm buffers do not drift, and body state is restored in a `finally`
    path. Added named, validated settings for epochs, learning rate, momentum
    and weight decay. Defaults are 5 epochs, learning rate 0.01, momentum 0.5
    and weight decay 0.0, following the official repo's [argument defaults](https://github.com/jhoon-oh/FedBABU/blob/master/utils/options.py#L15-L17)
    and [fine-tuning epoch option](https://github.com/jhoon-oh/FedBABU/blob/master/utils/options.py#L49-L51).
    The framework's existing `models.utils.train()` now accepts optional SGD
    parameter/settings and frozen-module arguments; all existing defaults are
    retained. Tests verify the entire body including BatchNorm buffers and the
    original client/server models remain unchanged, the head changes, all
    training samples are used, configured optimizer values reach SGD, and bad
    settings are rejected. All 31 focused FedBABU tests passed; four device
    tests passed. A Ditto regression run passed 10 tests but one unrelated
    existing assertion fails because the network requires an experiment seed
    before reaching its hierarchy guard. The personalized fine-tune method is
    invoked by the run after the repeated-round loop (step 5.5). Step 6.1 adds
    the resolved setting values explicitly to experiment dictionaries in
    `main.py`.
[x] 5.3 Added `Client.evaluate_fedbabu_personalized()` using the existing
    non-mutating `test()` path and `testloader`, plus
    `evaluate_fedbabu_personalized_clients()` to build a stable, structured
    final-round record with `model_role='fedbabu_personalized'`. It evaluates
    the per-client personal models and does not append metrics to
    `client_log.pkl`; output persistence is defined in step 5.4. The focused
    test verifies client/server/personal model states are unchanged and the
    result carries the client/server identity and model-role metadata. All 32
    focused FedBABU tests passed.
[x] 5.4 Added `constants.Logs.FEDBABU_PERSONALIZED_CLIENT_LOG` with the
    dedicated base name `fedbabu_personalized_client_log` and
    `build_fedbabu_personalized_log()` using the repository's existing
    `write_structured_log()` envelope (`schema_version` plus `records`). A
    temporary-file round-trip test verifies the saved pickle preserves the
    personalized model role and metrics while the name remains distinct from
    `client_log`. No evaluation calculations or general log semantics changed.
    Step 5.5 connects this writer to the final evaluation record.
[x] 5.5 Connected finalization in `FederatedNetwork.run_simulation()` after
    the repeated-round loop and before existing log persistence. FedBABU now
    creates the personal copies from the final server, fine-tunes each head,
    evaluates through the existing client test path, and writes the versioned
    payload to `fedbabu_personalized_client_log.pkl` under the selected log
    directory. Ordinary client/server/evaluation logs remain on their existing
    paths and formats; the loop body is unchanged. The integrated finalizer
    passed a focused CPU test and all 34 FedBABU tests passed. Compilation and
    `git diff --check` passed; a full `FederatedNetwork` simulation is still
    pending step 7.3.

    Check: focused tests prove fine-tuning changes only the personal head,
    evaluation is non-mutating, the shared/server model stays unchanged, and
    the personalized log remains separate from ordinary client metrics.

### 6. Add configuration and experiment handles

Files: `main.py`, `federated_network/network.py` only for constructor/config
validation (not the simulation loop), `tests/test_fedbabu_strategy.py`.

[x] 6.1 Add shared `fedbabu_head_finetune_*` settings to `main.py` next to
    the existing Ditto/FairFedDrift settings. Their explicit values match the
    validated defaults in `strategy/FedBABU/utils.py::resolve_fedbabu_parameters`;
    the FedBABU handle remains disabled until step 6.2.
[x] 6.2 Added fully commented FedBABU constructor/run handles in the MNIST,
    Fashion-MNIST, CIFAR-10, CIFAR-100 and Tiny ImageNet sections. They reuse
    each section's dataset, drift and round settings, shared FedBABU recovery
    parameters, and established per-strategy output directories. The shared
    parameters select FedBABU as both recovery and base aggregation method and
    include the validated head fine-tuning settings. AST/source checks verify
    all five handles remain disabled, so they add no constructors to normal
    runs.
[x] 6.3 Audited the parsed `main.py`: it currently has 16 active
    `FederatedNetwork(...)` constructor call sites and one active
    `run_simulation(...)` call (FairFedDrift on MNIST). All five FedBABU
    constructor and run blocks are comments, so they create no networks. No
    experiment selection was changed. Before any full `main.py` validation,
    temporarily isolate the intended small FedBABU experiment so other active
    constructors do not load datasets or allocate models.

    Check: every planned handle constructs the FedBABU strategy and routes
    through the ordinary simulation entry point.

### 7. Validate incrementally

[x] 7.1 Ran `py -3.10 -m unittest discover -s tests -p
    test_fedbabu_strategy.py`: all 36 tests passed. Coverage includes model
    partitioning, fixed-head warm-up/round/drift behavior, body-only server
    aggregation and client distribution, shared-head initialization,
    personalized head-only training/evaluation/finalization, and separate log
    persistence. This is focused unit coverage, not a full simulation.
[x] 7.2 Ran the available regression suites. Device (4), staged evaluation
    (4) and model-distance (5) tests passed. The FairFedDrift suite ran 99
    tests: 94 passed and 5 failed/errored due to a stale drift-function
    signature, FairFedDrift main-handle assertions not matching the current
    MNIST handle, and missing `ucimlrepo`/`random_utils` imports. Ditto strategy
    and simulation suites could not import because `ucimlrepo` is unavailable.
    These unrelated configuration/API/dependency issues were recorded without
    modifying existing strategies or experiment selection. CUDA-only execution
    is not validated in this CPU environment.
[x] 7.3 Ran a synthetic MNIST-shaped, full-participation CPU simulation with
    two clients and two training rounds through the real
    `FederatedNetwork.run_simulation()` loop. A controlled label-swap drift
    fixture exercised onset and end transitions; the check found that
    `change_server_aggregation_strategy()` fell through to FedAvg for FedBABU.
    Added an explicit FedBABU strategy-switch case and a regression test. The
    rerun passed: FedBABU remained selected across drift, shared client/server
    heads remained fixed, personal heads were fine-tuned, and ordinary plus
    personalized logs were written. Dataset loading/splitting and plotting
    were stubbed, so this is an end-to-end synthetic smoke test rather than a
    real-dataset experiment. All 37 focused FedBABU tests pass after the fix.
[x] 7.4 Ran matching two-client, two-round synthetic CPU drift simulations for
    FedAvg and FedEx through the real network loop. Both retained their
    configured strategy through drift onset/end and wrote the ordinary client,
    server and evaluation logs. Dataset loading/splitting and plotting were
    stubbed as in the FedBABU smoke test; this verifies compatibility on the
    controlled path, not benchmark behavior on downloaded datasets.
[x] 7.5 Recorded validation boundaries: `ucimlrepo` and `random_utils` are
    absent in this environment; PyTorch is CPU-only; no downloaded-dataset or
    GPU benchmark was run. The end-to-end checks used synthetic MNIST-shaped
    data and patched loading/splitting/plotting. Five FairFedDrift regression
    tests fail/error on existing API/handle assumptions or missing imports,
    and Ditto test modules cannot import without `ucimlrepo`. FedBABU supports
    only the explicitly mapped final linear heads in
    `strategy/FedBABU/utils.py::FEDBABU_HEAD_MODULE_BY_MODEL_TYPE`; unknown or
    incompatible architectures are rejected rather than guessed.

## Out of scope for the first integration

- Changing the main simulation loop, FedAvg, FedEx, or existing model
  definitions solely to match upstream naming.
- FedBABU variants that train the head on server data, alternative body/head
  update modes, and unrelated robust aggregation methods.
- Changing dataset labels, drift specifications, or the framework's metric
  calculation functions.
- Treating the end-of-training personalized head evaluation as an online
  per-round adaptation method; that would be a separate algorithmic extension.
