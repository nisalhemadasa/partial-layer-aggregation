FairFedDrift integration plan — adaptation to PaLA's client-group drift
Revised: 2026-09-24
Status: Steps 1.1-1.4, 2.2-2.5, 3.1-3.6, 4.1-4.6, 5.1-5.4, 6.1-6.3a, 7.1-7.6, 8.1-8.2, 8.3a and 8.4 complete; step 2.1 cleanup complete.
Unused sensitive-group metadata helpers and their dedicated tests have been removed.
Python 3.10 imports the drift/server modules; full network import requires ucimlrepo.
Full-budget PaLA scenario validation and external notebook metric validation remain pending.
The matched 15-round MNIST-subset pilot for scenarios A/B/C and FedAvg, FedEx,
Oracle and FairFedDrift passed on 2026-09-24; see step 8.3b for scope and limits.
FairFedDrift now accepts the framework's CPU or CUDA device selection; actual
CUDA execution remains to be validated on the user's GPU server.

Goal
Add FairFedDrift as another strategy in the current framework, like Oracle,
FedEx and Ditto. Reuse the existing files, functions, models and PyTorch stack.
Keep the main simulation loop in federated_network/network.py unchanged.

Target: PaLA's client-group distributed concept drift, where subsets of clients
undergo shared label-mapping changes. No sensitive groups are required within
clients. Use one whole-local-dataset loss per client and one shared loss-increase
threshold, with a separate previous-loss reference for each client. Retain dynamic
cluster creation, assignment, merging and historical training from the upstream
implementation. Describe results as a single-group FairFedDrift adaptation
(FedDrift-style baseline), not unchanged fairness-aware FairFedDrift.

Ground-truth drift identities may drive simulation, Oracle and offline evaluation,
but must never drive this strategy's detector, assignments, merges or resets.
Configuration now requires fairfeddrift_loss_threshold and rejects old threshold keys.
Next implementation task: step 8.3b, run the planned PaLA scenario comparisons; the short real-data MNIST smoke in step 8.3a passed, and main.py's normal experiment selection remains unchanged.
Fidelity audit: see FairFedDrift_fidelity_review.txt (2026-09-23). The identified
merge-evaluation scope mismatch was fixed in step 3.4a; runtime equivalence remains unverified.
Final experiments use drift configurations specified and controlled in main.py.
Synthetic fixtures stay under tests/ and are never used by experiment handles.

Selected history policy: retain the latest 100 communication rounds by default.
fairfeddrift_window counts rounds in this integration, unlike upstream's timestep
window. This does not change the total experiment duration. None remains an
explicit unbounded override, but is not the default. History storage/eviction
helpers are implemented in step 2.4; runtime integration remains pending step 5.3.

How to use this plan
Complete one checkbox at a time. Each task names the existing code to reuse
and a small check. Update DEV_PROG.md after each development step, recording
what passed and what remains unverified.

1. Add the strategy entry point

[x] 1.1 Add FAIRFEDDRIFT to constants.py::RecoveryAlgorithm.
    Verified: constant imports correctly; existing strategy names are unchanged.

[x] 1.2 Create strategy/FairFedDrift/__init__.py and fairfeddrift.py.
    Follow strategy/Oracle/oracle.py and strategy/FedEx/fedex.py:
    FairFedDrift class, strategy_name, aggregate_models() and aggregator_fn().
    Export it through strategy/__init__.py in the existing style.
    Verified: Python 3.11 package import and factory return the correct instance.
    Aggregation explicitly raises NotImplementedError until implemented.

[x] 1.3 Add selection in federated_network/server.py::server_fn().
    Reuse Server and its existing dataset-specific model creation.
    Verified: isolated actual factory/Server initializer creates the correct
    MNIST CPU model and strategy for FairFedDrift, FedAvg, Oracle, FedEx and Ditto.
    Full server-module import remains blocked by missing SciPy on Python 3.11.

[x] 1.4 Revise resolve_fairfeddrift_parameters() in
    strategy/FairFedDrift/fairfeddrift.py, retaining Ditto's validation pattern.
    Replace fairfeddrift_threshold_privileged/fairfeddrift_threshold_unprivileged
    with one required fairfeddrift_loss_threshold (finite, nonnegative, not bool).
    Reject obsolete threshold keys with a migration message instead of silently
    ignoring them. Require no sensitive-group metadata or missing-group check.
    Keep defaults: fairfeddrift_window=100 (communication rounds),
    fairfeddrift_rounds_per_timestep=1, fairfeddrift_seed=42.
    Retain validate_fairfeddrift_setup() checks for a configured CPU or CUDA
    device, full participation and a single hierarchy level. Update
    tests/test_fairfeddrift_strategy.py.
    Check: explicit zero/positive thresholds pass; missing, negative, nonfinite
    and obsolete settings fail. Input dictionaries remain unchanged.
    Verified before metadata cleanup: all 13 focused configuration/data tests passed on Python 3.11,
    including zero/positive thresholds, invalid values, obsolete-key migration
    and unchanged input settings. Experiment-construction invocation is pending.
    Follow-up: the original CPU-only restriction was removed on 2026-09-24 after
    auditing FairFedDrift's data, training, loss, merge and aggregation paths;
    each uses CPU snapshots or the existing configured-device APIs. CUDA setup
    validation now accepts torch.device('cuda'), but real GPU execution remains
    unverified until the remote-host smoke run.

2. Reuse ordinary client data and the existing drift pipeline

[x] 2.1 Reuse data/dataset_loader.py::load_datasets() and data/utils.py::
    split_iid_dataset(), split_noniid_dataset() and convert_dataset_to_loader().
    Cleanup: removed set_dataset_groups()/get_dataset_groups() from data/utils.py
    and removed their dedicated tests/test_fairfeddrift_data.py. These unused
    helpers belonged to the original sensitive-group design. Existing loaders,
    partitioners, class labels and (inputs, labels) batches remain unchanged.
    No artificial privileged/unprivileged labels will be generated.

[x] 2.2 Add a small multi-client fixture with ordinary (inputs, labels) samples:
    stationary clients and clients with two different label-swap patterns.
    Keep true drift identities in the test/evaluation harness only.
    Implemented tests/fairfeddrift_fixtures.py::make_client_drift_fixture():
    six clients (two per concept), twenty samples each, independent before/after
    datasets, with drift identities returned separately to the test harness.
    Verified: four fixture checks plus nine configuration checks passed on
    Python 3.11, covering mappings, counts, two-field batches, isolation and
    repeatability. Detector integration and empty-data rejection await step 3.1.

[x] 2.3 Reuse the existing client-level drift path:
    federated_network/utils.py::apply_drift_to_clients() calls the drift code
    under drift_concepts/drift.py::apply_drift(). Keep drift transformations
    in drift_concepts/ and supporting helpers in drift_concepts/utils.py.
    Use drift_specs supplied by main.py to select clients, swap pairs and timing;
    do not hard-code fixture patterns in runtime code or require within-client masks.
    Verified with tests/test_fairfeddrift_drift.py on Python 3.10: configured
    swap pairs/timing, unchanged stationary clients/features, loader refresh,
    no repeated swaps and no Oracle reassignment. Existing drift functions stay
    unchanged; start/end handlers now preserve FairFedDrift strategy/client state.
    Simulation client.drift_id remains necessary for generating configured swaps;
    future detector/merge APIs must not consume it (to verify in steps 3 and 4).
    All 14 FairFedDrift tests passed. Broader discovery passed 26 tests but two
    Ditto modules failed import due to missing ucimlrepo. Main.py simulation
    validation is also pending that dependency and remaining algorithm wiring.

[x] 2.4 Retain timestep data references/snapshots through data/utils.py helpers.
    Preserve past labels when later drift changes current data.
    Tag batches with their arrival communication round. At round r, retain
    arrivals in [max(0, r-window+1), r]; reuse within a timestep must not refresh
    a batch's age or duplicate its samples. Default window is 100 rounds.
    Check: after 101 arrivals, evict the oldest; test multiple rounds per
    timestep and an explicit unbounded override. Specify warm-up indexing.
    Implemented data/utils.py::DatasetSnapshot and
    strategy/FairFedDrift/utils.py::ClientDataHistory (one instance per client).
    Snapshots materialize only selected tensor samples as detached CPU copies;
    item reads return copies. Random transformations are frozen at capture time.
    Warm-up is index 0; the first subsequent training round is index 1. Map these
    indices explicitly to the existing loop when wiring; do not move loop events.
    advance(round) expires arrivals even without new batches and returns expired
    round IDs; call it before assignment cleanup/add in the later integration.
    Repeated add for the same arrival does not resnapshot or extend retention.
    All 19 focused tests passed on Python 3.10, including five history checks
    for snapshot isolation, 101 arrivals, object release, timestep reuse, clock
    jumps, unbounded history and invalid inputs. No simulation wiring yet.
    Cross-arrival sample-version deduplication and cluster assignment cleanup
    remain in steps 5.1/5.3. Consumers must release their expired snapshot references.

[x] 2.5 Define current local data for detection separately from retained history.
    Reuse Client.sample_data() and data/utils.py loaders where suitable; never
    use client.testloader or the final test set to choose clusters/thresholds.
    Check: all candidate models see the same current decision samples, while
    historical training and merge checks use their own retained data views.
    Implemented strategy/FairFedDrift/utils.py::prepare_fairfeddrift_decision_loader().
    It snapshots client.trainloader.dataset after drift/sampling, uses the existing
    batch size and returns a non-shuffled loader retaining the final partial batch.
    Build once per decision event and reuse for every candidate. No resampling,
    testset/testloader access or history append occurs. Missing/empty training
    data fails clearly. This follows the existing Subset-based Client.sample_data()
    contract; custom sampling/collation pipelines are not supported by this helper.
    All 22 FairFedDrift tests passed on Python 3.10, including three new checks
    for fixed candidate inputs, historical-label isolation and invalid sources.
    Runtime invocation remains pending step 4.3; model evaluation is step 3.1.

3. Implement only the FairFedDrift-specific decisions

Keep the main algorithm and its state in strategy/FairFedDrift/fairfeddrift.py.
Use strategy/FairFedDrift/utils.py for algorithm-specific helper calculations.
Do not introduce a separate experiment runner, model stack or logging system.

[x] 3.1 Calculate one sample-mean local loss per client/candidate model.
    Reuse models/utils.py::test() on current local decision loaders where its
    loss convention fits; otherwise add only the needed scalar-loss helper.
    Check: a hand-calculated example matches; evaluation restores model mode.
    Reject empty decision datasets clearly instead of returning NaN; validate
    the no-sensitive-metadata path with the fixture from step 2.2.
    Implemented strategy/FairFedDrift/utils.py::evaluate_fairfeddrift_loss().
    Existing test() averages batch means; leave it unchanged and use summed
    cross-entropy divided by actual sample count, including partial batches.
    Reuse the prepared step 2.5 loader across candidates. Use get_device() for
    inputs, no gradients, and restore all module modes even on evaluation failure.
    Reject empty/nonfinite losses. All 27 FairFedDrift tests passed on Python 3.10,
    including five loss checks for numerical averaging, model/gradient/buffer
    preservation, failure restoration and ordinary-label client drift fixtures.
    Runtime invocation remains pending step 4.3; selection/state follow in 3.2/3.3.

[x] 3.2 Select the lowest-loss acceptable model using fairfeddrift_loss_threshold.
    Candidate loss <= that client's previous reference loss + threshold passes.
    If all fail, create a cluster and retain the lowest failing loss as the next
    reference, following the upstream scalar version. Never pool clients' losses
    into one network-wide detector. The threshold is an absolute loss difference.
    Check: equality passes, ties are deterministic and each client's reference
    updates independently. Initial reference is 1000, as in upstream FedDrift.
    Implemented FairFedDrift.select_model(client_id, candidate_losses, loss_threshold).
    Returns (selected cluster ID, next reference loss); None as the ID requests
    cluster creation in step 3.3. Equal losses choose the first active candidate
    in mapping order. The minimum loss suffices because all candidates for a
    client share the same acceptance bound. Invalid inputs leave references intact.
    All 32 FairFedDrift tests passed on Python 3.10, including five selection
    checks for initial reference, boundaries, deterministic ties, failed-selection
    references, client isolation and invalid inputs. Runtime wiring is pending.

[x] 3.3 Track cluster models, client assignments and previous scalar losses.
    Reuse select_model() and its per-client previous_losses mapping from 3.2.
    Create a copied model when no existing model is acceptable.
    Reuse existing model architectures; set_parameters() remains the existing
    weight-download path for later wiring. It does not construct an independent
    model, so use deepcopy here without a redundant parameter reload.
    Check: new models have separate tensors and stable cluster IDs.
    Implemented initialize_clusters() and assign_client() in fairfeddrift.py,
    with cluster_models/client_assignments registries and increasing cluster IDs.
    Initialization copies the supplied model once. A failed threshold comparison
    copies the first currently active model, even when its ID is no longer zero.
    Candidate losses may cover a frozen subset of active models so new clusters
    do not alter other clients' candidate sets within the same decision event.
    Actual candidate freezing and shared server state remain step 4 wiring.
    All 37 FairFedDrift tests passed on Python 3.10, including five cluster checks:
    independent parameters/buffers, reassignment, stable IDs after simulated
    retirement, invalid inputs and rollback after model-copy failure.
    Merge retirement/history updates remain steps 3.5/5; no simulation wiring yet.

[x] 3.4 Calculate merge eligibility from historical whole-client losses.
    Check: compare all required client loss pairs and exclude current-timestep
    data. This calculation is separate from the existing weight-L2 diagnostics.
    Use the same single threshold for the scalar loss-difference gate and retain
    the upstream complete-linkage updates. Replacing detection alone is not enough.
    Rechecked upstream fed_drift.py at the pinned revision. Added
    calculate_fairfeddrift_merge_distances() and update_fairfeddrift_complete_linkage()
    in strategy/FairFedDrift/utils.py. Inputs are active models, advanced
    ClientDataHistory objects, client -> arrival-round -> cluster assignments,
    and the first round of the current timestep. Only retained earlier arrivals
    are evaluated, concatenated separately for each client/cluster combination.
    Scalar distance is max(0, max(Lij)-min(Lii), max(Lji)-min(Ljj)); this covers
    all cross-client pairs. Equality at the shared threshold is eligible.
    Return symmetric distances and historical sample counts. Missing history,
    self-pairs and rejected pairs use infinity rather than upstream's 1000 sentinel.
    Complete linkage keeps max(distance(A,C), distance(B,C)) after merging A/B;
    it does not recompute distances using the merged model. Helpers do not mutate
    assignments/history or merge models. Caller must advance histories to the
    current round; current-timestep arrivals are excluded even over multiple rounds.
    All 42 FairFedDrift tests passed on Python 3.10, including five merge checks
    for all-pairs gating, per-client sample means, current/expired-data exclusion,
    missing/invalid history and complete linkage. Runtime history construction,
    version deduplication and merge execution remain pending steps 3.5/4/5.

[x] 3.4a Audit correction: only evaluate models with retained historical
    data during merge comparison. Skip evaluation entirely with fewer than two
    history-bearing clusters. Upstream merge() only compares pairs when both
    have retained data, so history-free models need no forward pass.
    Implemented in strategy/FairFedDrift/utils.py. Counts and infinite distances
    are prepared first; if fewer than two clusters retain data, return without
    any model forward pass. Otherwise evaluate only history-bearing models.
    Added raising-model checks for no-history, one-history and history-free models
    alongside two eligible clusters. All 50 FairFedDrift tests passed on Python
    3.10. Merge eligibility semantics and history counts are unchanged.

[x] 3.5 Merge compatible clusters and update assignment history.
    Check: merge order, copied weights, retained sample counts and retired IDs
    match a small worked example; no client references a removed cluster.
    Implemented FairFedDrift.record_assignment() and merge_clusters(), with a
    client/arrival assignment_history mapping. Duplicate arrival recording does
    not overwrite the original assignment. Runtime callers must record each new
    retained arrival and supply every client history to merging.
    Reuse step 3.4 distances/counts and complete linkage; select the nearest finite
    pair, breaking ties in active order. Merge into a new increasing ID, relabel
    current assignments and all retained arrivals, and remove retired models.
    Current-timestep arrivals are relabeled but excluded from weights/comparisons.
    Prune expired assignment references against supplied retained histories.
    Stage the whole merge pass before committing; copy/load failures leave model
    registries, assignments, previous losses and the next ID unchanged.
    Added average_fairfeddrift_parameters() in strategy/FairFedDrift/utils.py,
    following Ditto's existing weighting convention and reusing set_parameters().
    Float/complex tensors use sample weighting. Integer/bool buffers use the
    largest contributor at each pairwise merge (first on ties); this is an
    explicit framework buffer policy. Existing Ditto/FedAvg code is unchanged.
    All 48 FairFedDrift tests passed on Python 3.10, including six execution
    checks for repeated merges, counts/weights, retirement and assignment updates,
    complete-linkage blocking, rollback, expiry and invalid averaging inputs.
    Runtime arrival recording/history wiring remains pending in steps 4/5.

[x] 3.6 Add sample-count-weighted aggregation for FairFedDrift.
    Reuse average_fairfeddrift_parameters() added for cluster merges in 3.5;
    Implemented FairFedDrift.aggregate_models(server_model, uploads, sample_counts).
    Reuse the validation/weighting pattern in strategy/Ditto/ditto.py::
    Ditto.aggregate_models(). If extracting a shared weighted helper, place it
    in utils.py and verify Ditto behavior remains identical.
    Keep strategy/FedAvg/fedavg.py's equal averaging unchanged.
    Preserve upstream weighting for this adapted baseline and disclose that PaLA
    uses uniform averaging. A uniform-weight variant would be a separate ablation.
    Validate identical upload/count IDs, nonempty inputs, positive integer counts,
    exact ordered state keys and server tensor shapes/dtypes before mutation. Reuse
    average_fairfeddrift_parameters() then existing models/utils.py::set_parameters().
    Preserve the explicit largest-contributor integer/bool-buffer rule (first on
    ties). FedAvg and Ditto remain unchanged. Although the aggregation rule follows
    the upstream baseline, this weighting differs from PaLA's uniform extractor
    averaging, as recorded above.
    Check passed: values 0 and 4 with counts 1 and 3 aggregate to 3; also tested
    equal counts, single-client copies, buffers and invalid inputs. All 55 focused
    FairFedDrift tests passed on Python 3.10. Server/client flow still needs to
    supply the right cluster's uploads and counts in step 4.5.

4. Connect to the existing client/server flow

[x] 4.1 Trace two small rounds through the existing functions before wiring:
    client_initial_training() -> model_aggregation() -> diagnostics/evaluation
    -> apply_drift_to_clients() -> train_client_models().
    Identify initialization, timestep decisions and the final aggregation.
    Check: each operation fits an existing helper boundary; do not move the loop.
    Connect resolve_fairfeddrift_parameters() and validate_fairfeddrift_setup()
    at the earliest construction/configuration boundary before dataset loading.
    Use the existing device selection and pass its resolved device to validation.
    Code trace (inspection only): main.main() calls configure_device() before
    constructing FederatedNetwork. The constructor currently resolves Ditto
    before load_datasets(); add FairFedDrift resolution/setup validation at this
    same pre-data boundary using server_tree_layout, client_select_fraction and
    device_utils.get_device(). FairFedDrift's supported setup requires one flat
    server and full participation. Do not validate during import-time setup.

    Round flow: constructor builds clients/servers and link_clients_to_servers()
    assigns each client.parent_server_id. run_simulation() first calls
    client_initial_training(), which samples client data and trains local models.
    Loop round 0 then aggregates those warm-up uploads, distributes the result,
    and evaluates. For later rounds, handle_drift_for_round() changes drift state;
    aggregation/distribution occurs before apply_drift_to_clients(), which applies
    main.py's drift and refreshes each client's trainloader. train_client_models()
    then resolves the server by parent_server_id, downloads its parameters and
    calls Client.fit(). Thus the first detector event is after warm-up aggregation
    and current-data refresh, before that round's local training. The final loop
    iteration aggregates/evaluates but intentionally performs no new drift or fit.

    Wiring boundaries: initialize shared learned cluster state with the flat leaf
    server and ensure it refers to the live round-0 model (not a stale copy) in
    step 4.2; perform one frozen-candidate decision pass for all clients before
    train_client_models() resolves per-client servers in 4.3; retain ordinary
    client.model training through Client.fit()/models.utils.train() in 4.4; route
    tagged uploads and local sample counts by the cluster used in that training
    round through Server.train()/model_aggregation() in 4.5. Existing loop order
    already provides the required hooks; no loop edit is planned.

    Verified by source trace only: network.py, train_client_models(), Client.fit(),
    client_initial_training(), link_clients_to_servers(), model_aggregation() and
    Server.train(). No network construction or simulation was run; importing the
    full network still requires missing ucimlrepo. The initial cluster/model
    ownership and round-0 aggregation interaction must be verified in 4.3/4.5.

[x] 4.2 Add a small learned-assignment linker alongside
    federated_network/utils.py::link_clients_to_servers(). Keep learned cluster
    IDs separate from server-list positions, because train_client_models() uses
    client.parent_server_id as an index into the flat server list. The linker
    validates the active cluster/server correspondence and complete assignments,
    rebuilds server.client_ids, and writes positional parent_server_id values.
    Regression checks cover non-contiguous IDs, relinking, inactive IDs, and
    incomplete mappings. Do not use link_clients_to_servers_by_drift_id() for
    learned decisions. The shared strategy and initial live-model ownership are
    established in 4.3a; dynamic cluster/server model binding remains for 4.5.
    This helper only handles client and server IDs.
    Validation: four focused tests passed on Python 3.10; git diff --check passed.

[x] 4.3a Establish one FairFedDrift strategy state per run from the first call to
    train_client_models(), which happens after round-0 warm-up aggregation and
    current-round data refresh. initialize_fairfeddrift_runtime() shares the
    existing strategy object across the flat server list, copies the live
    server model into cluster 0 once, then binds the server to that registry
    model so subsequent aggregation updates the live cluster. Repeated calls
    preserve model state. Four focused Python 3.10 tests passed; git diff --check
    passed. No simulation-loop edits.

[x] 4.3b Add the pre-training FairFedDrift decision pass before the per-client
    loop in train_client_models(). The network constructor now resolves and
    validates FairFedDrift settings before dataset loading, then passes the
    resolved threshold through server_fn() into the shared strategy state.
    Snapshot each client's current sampled training subset once via
    prepare_fairfeddrift_decision_loader(), and evaluate every candidate model
    using that same per-client loader. Freeze candidate IDs/model references for
    the whole pass, compute all losses before assigning anyone, then update
    learned assignments from observed loss only. Newly created clusters are not
    candidates until the next pass. synchronize_fairfeddrift_servers() creates
    or reorders flat server slots to match active learned IDs, and the existing
    linker writes each client's positional parent_server_id. No simulation-loop
    edit. Two decision tests, all 65 FairFedDrift tests and four device tests
    passed on Python 3.10; git diff --check passed. Client training and upload
    aggregation for the learned assignments remain pending steps 4.4-4.5.

[x] 4.4 Reuse the existing client download and local-training path. The learned
    parent_server_id is resolved before model download in train_client_models(),
    and set_parameters() copies that cluster server's state into client.model.
    Client.fit() already uses models/utils.py::train() before and after drift,
    but its active-drift dispatch omitted FAIRFEDDRIFT and therefore skipped
    training. Added FAIRFEDDRIFT to the existing ordinary local-training branch.
    Two focused tests verify assigned server weights arrive before fit and local
    training runs before, during and after drift. The framework warm-up remains
    independent local training before round-0 aggregation. All 67 FairFedDrift
    tests passed on Python 3.10; git diff --check passed. Upload grouping and
    weighted aggregation remain pending step 4.5.

[x] 4.5 Route tagged uploads to the learned cluster used during local training.
    train_client_models() records fairfeddrift_training_cluster_id immediately
    before Client.fit(). model_aggregation_fairfeddrift() filters by that tag,
    validates tags against active cluster servers and calls each server once
    with only its clients' state dictionaries. The initial untagged warm-up is
    accepted only for the sole cluster-0 server. Counts use len(client.local_trainset),
    matching Ditto's existing aggregation convention; FairFedDrift's strategy
    applies its weighted state helper through Server.train(). Empty clusters are
    skipped. Tests cover warm-up, weighted aggregation, swapped/stale parent
    positions, invalid routing and server dispatch. All 71 FairFedDrift tests
    passed on Python 3.10; git diff --check passed. No simulation-loop edits.

[x] 4.6 Align phase switching with recovery_method/base_aggregation_method.
    Reuse FederatedNetwork's initial strategy selection and the existing drift
    handlers: base_aggregation_method runs before/after drift; recovery_method
    runs during drift. Setting both to FairFedDrift keeps that strategy active
    throughout. Preserve its strategy object when switching away and back.
    A single-server base (e.g. FedAvg/FedEx) and Oracle as base with FairFedDrift
    recovery are supported. Oracle's fixed servers are parked/restored around
    FairFedDrift's learned topology. FedRC remains incompatible; Oracle as the
    drift recovery over a FairFedDrift base is rejected because it needs a fixed
    ground-truth server layout not provided by the learned topology.
    Check: handler transitions are tested; Oracle server restoration/linking is
    tested. Full simulation and state persistence across actual training remain
    unverified until steps 5-8.

First working milestone: a small CPU run with window=1 round and one communication
round per data timestep. This is a restricted version; continue with step 5.

5. Add historical training using the same training functions

[x] 5.1 Build per-client loaders from retained data grouped by historical cluster.
    Added strategy/FairFedDrift/utils.py::build_fairfeddrift_history_loaders(),
    reusing ConcatDataset and data/utils.py::convert_dataset_to_loader(). It
    returns loaders and unique selected-sample counts per cluster. ClientDataHistory
    now interns exact captured input/label versions while preserving each arrival
    round as a separate assignment reference; expired arrivals release shared data
    only after their final reference expires. A repeated version assigned to one
    cluster is trained once there, but may contribute independently to another.
    This deduplication is the PaLA adaptation to its reused local datasets, unlike
    upstream's newly arriving timestep batches.
    Check: a single client's history assigned to two clusters yields both loaders;
    duplicate versions count once per cluster, changed labels remain distinct, and
    eviction releases pooled samples after their final retained arrival expires.
    The new loaders are helper-level only; runtime training integration is step 5.2.

[x] 5.2 Train retained history through the existing Client/models.utils.py path.
    Added Client.fit_fairfeddrift_history(): for each cluster represented in the
    client's retained history, deep-copy that cluster's server model, call the
    existing models.utils.py::train() on the cluster's history loader, and retain
    only detached parameter uploads plus their sample counts. Each call creates a
    separate temporary model; the existing train() routine creates a fresh local
    optimizer per call. The ordinary client.model is never used or mutated here.
    Check: distinct cluster models start from their own server parameters, local
    updates do not mutate server/client models or each other, upload tensors are
    detached copies, and inconsistent sample counts fail before training.
    This method is not called by the simulation orchestration yet; 5.3/5.4 connect
    these uploads to aggregation and per-round retained-history scheduling.

[x] 5.3 Route per-client/per-cluster history uploads through the existing
    server aggregation path. model_aggregation_fairfeddrift() now groups ordinary
    current-assignment uploads as before and adds each client's historical upload
    for the active cluster. Server.train() forwards both groups to the strategy,
    which validates each model/count mapping and aggregates all contributions by
    their respective positive sample counts. A cluster may receive only historical
    uploads; runs without history uploads retain the existing step-4.5 behavior.
    Client.fit_fairfeddrift_history() supplies exact unique-sample counts from its
    loader and clears old uploads before each call, including when history is empty.
    Inactive cluster references are rejected. The bounded history default is 100
    rounds; orchestration that refreshes these uploads and cleans expired
    assignment references and clearing old uploads are wired through runtime orchestration.
    No history tensors are stored in logs.
    Check: weighted current+history aggregation, history-only aggregation,
    per-cluster routing, exact counts, inactive cluster rejection and empty-history
    upload clearing are covered by tests.

[x] 5.4 Track data timesteps through existing orchestration helpers, allowing
    multiple communication rounds per timestep without rerunning detection.
    `federated_network/utils.py::prepare_fairfeddrift_timestep()` is called from
    `train_client_models()`: decisions run once per configured communication-round
    bucket, independent of drift ground truth; previous arrivals are merged before
    decisions, the current snapshot is stored once after assignment, and it is
    excluded from its own historical training. Client history advances on every
    round, expired assignment references are removed, and each sampled client's
    historical per-cluster uploads are refreshed for that round. The existing
    simulation loop in `federated_network/network.py` is unchanged.
    Check: two rounds per timestep reuse one assignment; drift state changes do
    not trigger extra decisions; prior history is used by the next timestep;
    expired arrivals disappear. All 88 FairFedDrift tests and 4 device tests pass
    on Python 3.10; `git diff --check` passes.
    Scheduling follows the framework's configured communication-round buckets.
    The local simulation keeps its ordinary final aggregation/evaluation behavior;
    it does not add upstream's separate final test-only timestep.

6. Reuse evaluation, diagnostics and log saving

[x] 6.1 Extend `server.py::server_hierarchy_evaluate()` using its Oracle branch
    as the reference for evaluating all active FairFedDrift cluster servers.
    Detect FairFedDrift from each server's active strategy, then reuse
    `Server.model_evaluate()` / `models/utils.py::test()` when the server has test
    data, or `Server.average_client_evaluation_results()` with the configured
    client-metric weighting otherwise. Empty servers retain Oracle's zero-value
    placeholder. A cached but inactive FairFedDrift strategy does not trigger the
    cluster path. The existing simulation loop and its result shape are unchanged.
    Check: all active cluster servers are evaluated, no-test-data weighting is
    forwarded, empty servers are handled, and inactive cached strategy state does
    not alter evaluation. Three focused tests and all 91 FairFedDrift tests pass
    on Python 3.10; `git diff --check` passes.

[x] 6.2 Check `utils.py::evaluate_clients_for_stage()` and
    `distance_metrics/distance_metrics.py::collect_model_distance_diagnostics()`
    against changing FairFedDrift assignments. The learned cluster ID is distinct
    from the flat server-list position: staged evaluation resolves the current
    position while its records also include `fairfeddrift_cluster_id`; distance
    diagnostics retain server ID/absolute ID and now include the learned cluster
    ID. `model_id='primary'` remains valid because each FairFedDrift server owns
    one ordinary model. Existing `ModelDistanceHistory` capture remains in
    `network.py` after aggregation and before distribution. Evaluation remains
    non-mutating; whole-model distance remains the square root of the sum of all
    included tensor squared differences.
    Check: regression tests reindex a learned cluster to flat slot 1, verify its
    model/cluster identity, preserve evaluation model state, and check a 3-4-5 L2
    distance. Thirteen evaluation/diagnostic tests and all 92 FairFedDrift tests
    pass on Python 3.10; `git diff --check` passes. No simulation-loop changes.

[x] 6.3 Keep FairFedDrift's standard performance logs in the framework's
    existing Oracle/FedEx formats. `client_log.pkl` is still a list by round of
    per-client `(loss, accuracy)` pairs written by `write_logs()`. `server_log.pkl`
    is still a list by round of flat-level server `(loss, accuracy)` pairs in the
    same nested layout returned for Oracle. FairFedDrift's server-list length can
    vary by round as clusters are created or merged; consumers must not assume a
    fixed cluster count. The additional `fairfeddrift_state_log.pkl` is
    strategy-specific metadata and does not replace either performance log.
    No PaLA metric calculations are added to these logs.

[x] 6.3a Add FairFedDrift-specific scalar records for per-candidate local losses,
    decision-snapshot sample counts, assignments, new clusters, merges, timestep/
    round IDs, active clusters, per-round historical-upload sample counts and
    resolved settings. `FairFedDrift.runtime_history` stores these events without
    retaining dataset snapshots or model tensors. The post-run
    `build_fairfeddrift_state_log()` uses `write_structured_log()` and the
    `constants.Logs.FAIRFEDDRIFT_STATE_LOG` name. `FederatedNetwork` retains the
    shared strategy reference so the log is written after the simulation loop
    even if another configured phase strategy is active at the end. Existing
    `client_log.pkl` and `evaluation_log.pkl` meanings are unchanged; no statement
    was added to the simulation loop.
    Check: event capture covers decision and repeated-timestep rounds; structured
    records pickle and reload with assignments, losses, counts, merges, and
    parameters. Five logging/scheduling tests and all 94 FairFedDrift tests pass
    on Python 3.10; `git diff --check` passes.

[ ] 6.4 Validate PaLA's group metrics in the separate evaluation notebooks.
    Do not add PaLA metric calculation functions to this repository for now;
    no Jupyter notebooks are present in this checkout. Use the existing
    `client_log.pkl` and `server_log.pkl` inputs, plus `fairfeddrift_state_log.pkl`
    when learned assignments or cluster-count history are needed. Confirm the
    notebook reader handles a variable number of server entries by round.
    Check the paper's accuracy dip (D=4), recovery R2/R10, convergence accuracy
    (last 5 rounds), AEQ with trailing 2-round accuracy smoothing, horizons of
    5 rounds and 11/20 rounds for MNIST/Fashion-MNIST and CIFAR-10. Group identity
    comes from experiment drift specifications for reporting only. Preserve
    explicit zero-denominator behavior and report accuracy alongside AEQ and
    model/runtime/communication/memory costs.

7. Add the main.py handles, like the existing algorithms

This is an explicit integration deliverable, not merely a strategy enum entry.
Use main.py's shared ditto_parameters and per-dataset Ditto/Oracle blocks as
references. Keep configuration inside main.py and retain its current structure.

[x] 7.1 Add shared `fairfeddrift_parameters` beside `ditto_parameters` in
    `main.py`, containing the required loss threshold, 100-communication-round
    history window, rounds per timestep and seed. The threshold is an explicitly
    provisional experimental starting value (0.1), not a paper-recommended or
    validated optimum; tune it using validation data before final experiments.
    Check: configuration matches the parameter resolver's required key/defaults
    and validates without running or constructing any experiment handle.

[x] 7.2 Add shared `fairfeddrift_recovery_parameters` in `main.py` with both
    `recovery_method` and `base_aggregation_method` set to
    `RecoveryAlgorithm.FAIRFEDDRIFT`. Merge in 7.1's shared settings and retain
    only the common FedAU alpha and FedEx alpha values consumed by generic network
    construction/dispatch. Do not include FedRC or Oracle cluster-count settings;
    FairFedDrift begins with the flat server layout supplied by its handle and
    learns its active cluster count dynamically.
    Check: both phases select FairFedDrift; resolved settings and supported
    single-level/full-participation CPU setup validation pass without building an
    experiment handle or loading datasets.

[x] 7.3 Add a disabled MNIST FairFedDrift block containing both a
    `fairfeddrift_fed_net = FederatedNetwork(...)` constructor and its
    `run_simulation()` call. Reuse the adjacent MNIST Ditto/Oracle dataset,
    partitioning, drift, simulation, round-count and full-participation settings;
    pass `fairfeddrift_recovery_parameters` and use the flat `[1]` initial server.
    Keep the constructor and call commented so existing experiment selection is
    unchanged. The dedicated paths are added in step 7.4.
    Check: source inspection confirms both statements are disabled and the
    configuration tests verify the shared parameters; no dataset is loaded.

[x] 7.4 Give the MNIST handle dedicated plot and log paths:
    `plots/swap/MNIST/saved_plots_fairfeddrift/` and
    `logs/swap/MNIST/saved_logs_fairfeddrift/`. Pass both to its commented
    `run_simulation()` call so ordinary and FairFedDrift-specific output is
    isolated from other algorithms. The handle remains disabled.
    Check: the paths are inside the disabled FairFedDrift block and distinct from
    the existing FedAvg, FedEx, Ditto and Oracle output paths.

[x] 7.5 Add matching disabled handles alongside the existing algorithm blocks
    for Fashion-MNIST, CIFAR-10, CIFAR-100 and Tiny ImageNet-200, following the
    neighboring Ditto constructors' dataset and round-count settings. Each uses
    one initial flat server, full participation, the shared FairFedDrift config,
    and distinct plot/log paths. The handles are configuration examples only;
    their presence does not establish support or successful runs on each dataset.
    MNIST, Fashion-MNIST and CIFAR-10 are PaLA evaluation datasets; CIFAR-100 and
    Tiny ImageNet-200 are framework extensions.
    Check: source-level tests find exactly one disabled constructor/run block per
    dataset, verify paths/configuration and flat/full-participation settings.
    Three main-config tests and `git diff --check` pass; no experiment ran.

[x] 7.6 Keep BOTH constructors and run calls disabled by default. Audited all
    active FederatedNetwork(...) and run_simulation(...) calls in main.py.
    All five FairFedDrift handles (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 and
    Tiny ImageNet-200) remain commented out. The audit found 15 executable
    network constructors and three executable run calls: CIFAR-10 FedEx
    (the current alpha list has one value), Oracle and FedAvg. Constructors
    load data even when their run call is commented, so main.py is not isolated
    for a FairFedDrift smoke run. Existing experiment selection was left intact.
    Check: source/AST audit confirms no FairFedDrift handle executes; do not run
    main.py for step 8.3 until only the intended small experiment is active.

8. Validate and document the completed integration

[x] 8.1 Audit and run the focused FairFedDrift unit tests across their existing
    modules rather than duplicating them in tests/test_fairfeddrift_strategy.py.
    Coverage includes whole-client scalar loss, no sensitive metadata,
    threshold boundaries, deterministic ties, per-client references, empty
    decision data, weighted aggregation, merge eligibility/linkage and retained
    assignment updates. The scalar rule follows the upstream-derived adaptation
    specified in steps 3.2-3.5.
    Check: `py -3.10 -B -m unittest discover -s tests -p "test_fairfeddrift*.py"`
    passed all 98 tests. No cases were duplicated; end-to-end runtime coverage
    was added and verified in step 8.2.

[x] 8.2 Add tests/test_fairfeddrift_simulation.py following the small CPU setup
    in tests/test_ditto_simulation.py. Its real FederatedNetwork loop uses small
    synthetic MNIST-shaped data, no ground-truth drift and deterministic mocked
    candidate/merge losses to exercise the learned split, history-based merge,
    three-round history eviction, two rounds per timestep, training, evaluation
    and structured/ordinary logs. With no ucimlrepo installed, the test stubs
    only that unused fetch import; it does not download or use a real dataset.
    Check: `py -3.10 -B -m unittest discover -s tests -p "test_fairfeddrift*.py"`
    passed all 99 tests, including the end-to-end case. The real-data main.py
    run remains pending step 8.3 and must be isolated first.

[x] 8.3a Run an isolated real-data MNIST FairFedDrift smoke using the shared
    drift configuration in main.py. Temporarily routed main() through only this
    handle, using ten clients, five rounds, one local epoch and the first 5,000
    MNIST training / 1,000 test examples; then removed the temporary route.
    Output was validated in a temporary directory: five strategy events and
    five evaluation records. The default FairFedDrift handle remains disabled,
    and unrelated experiment selection is unchanged. This is an integration
    smoke only, not a full-budget result or accuracy comparison. PGF export
    warned because MiKTeX lacks underscore.sty; PNG/PDF export continued.
    Check: the network loop ran on CPU from the cached MNIST files; outputs were
    inspected before temporary files were discarded. The wrapper supplied a
    stub for the unused ucimlrepo import, which is absent in Python 3.10.

[ ] 8.3b Run the full PaLA scenario comparisons: A/B/C with drift 1 swaps
    (1,2)/(3,4), drift 2 swap (5,7), stationary clients and time-varying
    memberships. Use identical architectures, data splits, seeds, training
    budgets and evaluation stages across methods; validate threshold choices
    separately from final test data. Preserve held-out test data and document
    differences from upstream datasets. The 100-round history covers essentially
    all of PaLA's 50/100-round runs; test eviction explicitly in a longer run.
    Measure extra work from historical multi-model training. Main.py must be
    isolated before any full run because it has unrelated active constructors
    and simulation calls.
    Pilot check completed: all 12 scenario/method runs completed 15 rounds on
    the first 1,000 cached MNIST training and 250 held-out test examples, with
    ten clients, one local epoch and seed 42. A/B used two drift states at
    rounds 6/10; C used three at rounds 6/10/11, from the existing commented
    scenario configuration. All runs produced 15 client records and staged
    evaluation records; FairFedDrift also produced 15 state events per scenario.
    Logs: logs/pala_fairfeddrift_scenario_pilot_15r/{A,B,C}/{fedavg,fedex,
    oracle,fairfeddrift}/. This checks wiring and output compatibility only;
    it is not a PaLA-budget comparison, a threshold calibration or an accuracy
    analysis. An initial 50-round CPU attempt showed substantially higher
    FairFedDrift runtime and was stopped; its partial outputs were discarded.
    Full-data 50/100-round runs and measured history eviction/load remain open.

[x] 8.4 Run existing test_device.py, test_model_distance_diagnostics.py,
    test_evaluation.py, test_ditto_strategy.py and test_ditto_simulation.py.
    Also check small FedAvg/Oracle runs for unchanged averaging/evaluation.
    Verified: 26 tests passed on Python 3.10. The three pilot FedAvg runs each
    retained one server evaluation; all three Oracle runs retained evaluations
    for its three flat cluster servers. Ordinary FedAvg remains unweighted.
    The unused ucimlrepo import was stubbed for the Ditto tests; no dependency
    was installed. See DEV_PROG.md for the exact checks and warning.
    After removing the CPU-only setup guard, the complete focused
    test_fairfeddrift*.py suite also passed (99 tests) on CPU. Its configuration
    test confirms a CUDA device is accepted without allocating GPU tensors;
    actual CUDA execution is still unverified locally.

[ ] 8.5 Update README.md and AGENTS.md with supported modes, main.py handle
    instructions and remaining limitations. Read the edited documents back.
    Check: claims match completed tests; no generated artifacts are committed.

References and algorithm details to preserve

Target paper: D:/publications_related/2026/ICSOC/submission/version_2/PaLA_ICSOC_2026.pdf
Section 3 distinguishes client groups from sensitive groups within clients.
Sections 5.1-5.3 specify drift scenarios, 50/100-round runs and evaluation metrics.

Upstream: https://github.com/teresalazar13/FairFedDrift
Previously inspected revision: 51dec5eb1dc46a0c77e0ac0aa59c5724a46fab9a
Key upstream files: federated/algorithms/drift/fair_fed_drift.py,
 fed_drift.py, GlobalModels.py; metrics/LossPrivileged.py,
 metrics/LossUnprivileged.py; datasets/image/ImageDataset.py.
Retain upstream attribution (README declares CC BY 4.0) for adapted code.

- Original FairFedDrift uses s=1/s=0 losses. This adaptation uses the shared
  fed_drift.py machinery with one loss over each client's local data and one
  threshold; it does not reproduce the original sensitive-group mechanism.
- Initial previous loss is 1000 per client. New clusters copy the first currently
  active model; it may no longer have ID zero after merges.
- Merge comparisons use historical whole-client losses, exclude the current timestep
  and use complete-linkage updates. Merged weights use historical sample counts.
- Historical assignments can make a client train several cluster models.
  Upstream defaults to unbounded history and ten rounds per timestep.
- Keep framework training/model choices. Do not import upstream TensorFlow,
  fixed-CUDA code or experiment orchestration. Use get_device()/configure_device().
- Use PaLA's client-group AEQ for reporting, not sensitive-group equal opportunity
  or predictive parity. Keep existing label-swap class metrics and sample counts.

Done means: the single-loss adaptation works without sensitive metadata through
existing client/server functions, the main.py handles exist, historical behavior
is verified, and regression checks
pass without changing the main simulation loop. Document the scalar-loss,
round-window and data-reuse adaptations, and the weighting difference from PaLA.
Actual CUDA execution on hardware, deep hierarchies, partial participation and
published-result reproduction remain separate follow-ups.
