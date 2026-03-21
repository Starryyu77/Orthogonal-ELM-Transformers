# OELM Pretraining V2

This directory contains the official V2 experiment line for OELM pretraining validation.

## Status

- `experiments/oelm-pretrain/` remains the legacy pilot directory.
- `experiments/oelm-pretrain-v2/` is the only directory for official V2 code and configs.

## Methods

- `baseline`: standard trainable transformer baseline.
- `qk_only`: freeze `attn_q` and `attn_k`.
- `qk_ffn`: freeze `attn_q`, `attn_k`, `mlp_in`, and `mlp_out`.

## Directory Guide

- `configs/methods.json`: method definitions and freeze policy.
- `configs/quick_eval_tasks.json`: Phase 2 task and seed definitions.
- `configs/broad_eval_tasks.json`: Phase 5 task and setting definitions.
- `scripts/pretrain_v2.py`: pretraining entrypoint for Phases 0, 1, 3, and 4.
- `scripts/evaluate_downstream_v2.py`: downstream evaluator for Phases 2 and 5.
- `scripts/update_resource_ledger.py`: `sacct`-backed resource rollup utility.

## Cluster Artifact Layout

Code mirror:

- `/projects/LlamaFactory/OELM-Pretrain/experiments/oelm-pretrain-v2/`

Artifact roots:

- `/projects/LlamaFactory/OELM-Pretrain/logs/oelm-pretrain-v2/`
- `/projects/LlamaFactory/OELM-Pretrain/outputs/oelm-pretrain-v2/`
- `/projects/LlamaFactory/OELM-Pretrain/results/oelm-pretrain-v2/`

## Naming Convention

Outputs use:

`phase=<phase>/method=<method>/seed=<seed>/run=<timestamp_or_jobid>`

This keeps pilot artifacts and official V2 artifacts separate.
