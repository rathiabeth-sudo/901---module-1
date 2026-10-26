# Module 1 full experiment queue (Windows PowerShell). CPU-only runtime on an i7-10610U: ~4 hours.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Run($argsList) {
    Write-Host ">>> $argsList"
    & $py @argsList
    if ($LASTEXITCODE -ne 0) { throw "failed: $argsList" }
}

Run @("-m", "src.preprocessing")

# Module 1 - main task: performance (matching) class from geometry images
Run @("-m", "src.train_baseline", "--dataset", "patch", "--task", "perf_class", "--model", "majority")
Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "perf_class", "--mode", "linear_probe")
foreach ($s in 42, 43, 44) {
    Run @("-m", "src.train_baseline", "--dataset", "patch", "--task", "perf_class", "--model", "cnn", "--seed", "$s")
    Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "perf_class", "--mode", "finetune", "--seed", "$s")
}

# Amendment 1: revised-config candidate + validation-only adoption rule; augmentation ablation (Module 1 report Sec. 8.2)
Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "perf_class", "--mode", "finetune", "--profile", "revised", "--seed", "42")
Run @("-m", "src.select_final")
Run @("-m", "src.train_baseline", "--dataset", "patch", "--task", "perf_class", "--model", "cnn", "--no-aug", "--seed", "42")
Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "perf_class", "--mode", "finetune", "--no-aug", "--seed", "42")

# Secondary task: 5G/6G band class from the same geometry images  (adopted profile = preregistered, see results/metrics/final_model.json)
Run @("-m", "src.train_baseline", "--dataset", "patch", "--task", "band_class", "--model", "majority")
Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "band_class", "--mode", "linear_probe")
Run @("-m", "src.train_baseline", "--dataset", "patch", "--task", "band_class", "--model", "cnn", "--seed", "42")
Run @("-m", "src.train_vit", "--dataset", "patch", "--task", "band_class", "--mode", "finetune", "--seed", "42")

# Radiation-pattern modality (RadPat polar plots): beamwidth class
Run @("-m", "src.train_baseline", "--dataset", "radpat", "--task", "hpbw_class", "--model", "majority")
Run @("-m", "src.train_vit", "--dataset", "radpat", "--task", "hpbw_class", "--mode", "linear_probe")
Run @("-m", "src.train_baseline", "--dataset", "radpat", "--task", "hpbw_class", "--model", "cnn", "--seed", "42")
Run @("-m", "src.train_vit", "--dataset", "radpat", "--task", "hpbw_class", "--mode", "finetune", "--seed", "42")

# Analysis stages (need the checkpoints above)
Run @("-m", "src.explainability")
Run @("-m", "src.evaluate")
Write-Host "ALL EXPERIMENTS COMPLETE"
