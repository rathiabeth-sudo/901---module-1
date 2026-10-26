#!/usr/bin/env bash
# Module 1 full experiment pipeline (Linux/macOS). Equivalent to scripts/run_experiments.ps1.
set -euo pipefail
PY=${PY:-python}

$PY -m src.preprocessing

$PY -m src.train_baseline --dataset patch --task perf_class --model majority
$PY -m src.train_vit --dataset patch --task perf_class --mode linear_probe
for s in 42 43 44; do
  $PY -m src.train_baseline --dataset patch --task perf_class --model cnn --seed $s
  $PY -m src.train_vit --dataset patch --task perf_class --mode finetune --seed $s
done

# Amendment 1 + augmentation ablation
$PY -m src.train_vit --dataset patch --task perf_class --mode finetune --profile revised --seed 42
$PY -m src.select_final
$PY -m src.train_baseline --dataset patch --task perf_class --model cnn --no-aug --seed 42
$PY -m src.train_vit --dataset patch --task perf_class --mode finetune --no-aug --seed 42

$PY -m src.train_baseline --dataset patch --task band_class --model majority
$PY -m src.train_vit --dataset patch --task band_class --mode linear_probe
$PY -m src.train_baseline --dataset patch --task band_class --model cnn --seed 42
$PY -m src.train_vit --dataset patch --task band_class --mode finetune --seed 42

$PY -m src.train_baseline --dataset radpat --task hpbw_class --model majority
$PY -m src.train_vit --dataset radpat --task hpbw_class --mode linear_probe
$PY -m src.train_baseline --dataset radpat --task hpbw_class --model cnn --seed 42
$PY -m src.train_vit --dataset radpat --task hpbw_class --mode finetune --seed 42

$PY -m src.explainability
$PY -m src.evaluate
echo "ALL EXPERIMENTS COMPLETE"
