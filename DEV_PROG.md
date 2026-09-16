# Development progress

Keep one brief dated entry per development step: change, validation, and any remaining limitation. Record completed work only; implementation plans remain in their separate files.

## 2026-09-16

- Updated `AGENTS.md` to match this repository: excluded FedCollab, marked Ditto as planned, and corrected inherited assumptions. Read back and verified.
- Created `FRAMEWORK_IMPROVEMENTS_INTEGRATION_PLAN.md` covering seven framework improvements. Read back and verified; implementation tracked separately from planning.
- Implemented PGF export alongside PNG/PDF, directory creation, optional strict errors, and figure cleanup. Line/bar exports and XeLaTeX compilation visually verified; failure handling checked with Matplotlib 3.4.3. Full simulation not run.
- Created this progress log and added the requirement to update it after every development step to `AGENTS.md`. Read back and verified.
- Implemented shared auto/cpu/cuda selection, consistent model/strategy placement, guarded CUDA setup, and equivalent full-precision training. Four regression checks and a synthetic two-client/two-round CPU simulation passed on PyTorch 2.8.0+cpu; CUDA hardware execution unverified. Updated device docs and Phase B status.
- Implemented reusable L2 model-distance history for primary and FedRC models, collected post-aggregation/pre-distribution and persisted as whole-model/layer logs. Four focused checks and a two-client/two-round integration run passed; future FedEx optimization can consume `model_distance_history` directly.
- Implemented staged local/global evaluation, separate structured logs, label-swap class metrics, and optional sample-weighted server reporting. Eight focused checks and a two-client/two-round CPU run passed; client_log remains post-training local metrics.
- Attempted the integrated CPU validation step. The focused device, distance, and evaluation checks passed (12 tests), but the end-to-end partial-participation run is blocked because Phase C was explicitly skipped and the simulation loop still assigns all clients each round; no integrated result is claimed.
- Ran the remaining integrated validation with full participation. The existing two-client/two-round CPU run produced evaluation, class-metric, legacy client/server, drift, and model-distance logs; all 12 focused regression checks passed. CUDA execution and larger strategy/dataset combinations remain unverified.
