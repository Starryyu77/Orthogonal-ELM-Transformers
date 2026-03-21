# OELM Pretraining Validation Plan V2

## Summary

- Keep three main methods in scope: `baseline`, `qk_only`, and `qk_ffn`.
- Treat all prior `A5000` runs under `experiments/oelm-pretrain` as pilot runs only.
- Require implementation correctness before interpreting model quality.
- Run official V2 jobs on `cluster02` with `gpu:pro6000`.
- Keep all V2 code and artifacts isolated from the legacy pilot directory.

## Layout

- `configs/`: method, task, and seed configuration.
- `scripts/`: training, evaluation, and resource accounting scripts.
- `audits/`: freeze audits and implementation validation notes.
- `reports/`: phase summaries and experiment writeups.
- `manifests/`: run manifests and machine-readable experiment metadata.

## Phases

1. Phase 0: implementation and freeze audit validation.
2. Phase 1: corrected TinyStories mini pretraining.
3. Phase 2: corrected quick probe evaluation on `sst2` and `ag_news`.
4. Phase 3: OpenWebText subset confirmation.
5. Phase 4: OpenWebText formal run.
6. Phase 5: broad downstream evaluation.

## Resource Policy

- All official V2 jobs must request `gpu:pro6000`.
- `sacct` is the source of truth for queue wait, elapsed time, and final job state.
- Each run must emit:
  - `freeze_audit.json`
  - `training_summary.json` or `summary.json`
  - `resource_usage.json`
  - `run_manifest.json`
- Cluster-wide accounting rolls up into `results/oelm-pretrain-v2/resource_ledger.csv`.
