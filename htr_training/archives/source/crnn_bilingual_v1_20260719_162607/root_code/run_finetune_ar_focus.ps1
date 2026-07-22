param(
    [string]$RunName = "bilingual_v4_ar70_en30_ft",
    [int]$Epochs = 8,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 2
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$trainManifest = `
    ".\data\processed\master_ctc\train_finetune_ar70_en30.csv"

$validationManifest = `
    ".\data\processed\master_ctc\val_all.csv"

$vocab = `
    ".\data\processed\master_safe\vocab.json"

$sourceCheckpoint = `
    ".\outputs\checkpoints\bilingual_v4_master_ctc\best_cer.pt"

$initCheckpoint = `
    ".\outputs\checkpoints\finetune_initializers\ar70_en30_epoch22_init.pt"

$checkpointDirectory = `
    Join-Path ".\outputs\checkpoints" $RunName

$metricsPath = `
    ".\outputs\metrics\$RunName.csv"

$required = @(
    ".\scripts\23_train_crnn_resumable.py",
    ".\scripts\36_legacy_train_crnn_dual_gpu.py",
    ".\scripts\41_prepare_finetune.py",
    ".\data\processed\master_ctc\train_all.csv",
    $validationManifest,
    $vocab,
    $sourceCheckpoint
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required path: $path"
    }
}

if (Test-Path $checkpointDirectory) {
    throw @"
The output directory already exists:

$checkpointDirectory

Use a different -RunName or remove the old directory first.
"@
}

$prepareArguments = @(
    ".\scripts\41_prepare_finetune.py",
    "--train-manifest",
    ".\data\processed\master_ctc\train_all.csv",
    "--vocab",
    $vocab,
    "--source-checkpoint",
    $sourceCheckpoint,
    "--output-manifest",
    $trainManifest,
    "--init-checkpoint",
    $initCheckpoint,
    "--stats-output",
    ".\outputs\finetune_ar70_en30_distribution.csv",
    "--rows",
    "60000",
    "--arabic-ratio",
    "0.70",
    "--learning-rate",
    "1e-5",
    "--weight-decay",
    "1e-4",
    "--seed",
    "42"
)

Write-Host ""
Write-Host "Preparing balanced manifest and initializer..."

python @prepareArguments

if ($LASTEXITCODE -ne 0) {
    throw "Fine-tuning preparation failed."
}

$resumeCheckpoint = $initCheckpoint

for ($epoch = 1; $epoch -le $Epochs; $epoch++) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Fine-tuning epoch $epoch of $Epochs"
    Write-Host "========================================"

    $trainingArguments = @(
        ".\scripts\23_train_crnn_resumable.py",
        "--train-manifest",
        $trainManifest,
        "--validation-manifest",
        $validationManifest,
        "--run-name",
        $RunName,
        "--vocab",
        $vocab,
        "--resume",
        $resumeCheckpoint,
        "--epochs",
        "$epoch",
        "--batch-size",
        "$BatchSize",
        "--learning-rate",
        "1e-5",
        "--image-height",
        "64",
        "--max-image-width",
        "2560",
        "--hidden-size",
        "256",
        "--dropout",
        "0.1",
        "--num-workers",
        "$NumWorkers",
        "--gradient-clip",
        "5.0",
        "--accumulation-steps",
        "1",
        "--early-stop-patience",
        "3",
        "--lr-factor",
        "0.5",
        "--lr-patience",
        "1",
        "--lr-threshold",
        "0.001",
        "--lr-cooldown",
        "0",
        "--min-lr",
        "1e-6",
        "--seed",
        "42",
        "--output-root",
        ".\outputs",
        "--no-multi-gpu"
    )

    python @trainingArguments

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed during epoch $epoch."
    }

    $latestCheckpoint = `
        Join-Path $checkpointDirectory "latest.pt"

    if (-not (Test-Path $latestCheckpoint)) {
        throw "Latest checkpoint was not created."
    }

    $epochCheckpoint = Join-Path `
        $checkpointDirectory `
        ("epoch_{0:D2}.pt" -f $epoch)

    Copy-Item `
        $latestCheckpoint `
        $epochCheckpoint `
        -Force

    Write-Host "Preserved: $epochCheckpoint"

    $stateJson = python -c `
        "import sys,json,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps({'epoch':int(c.get('epoch',0)),'cer_bad_epochs':int(c.get('cer_bad_epochs',0))}))" `
        $latestCheckpoint

    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the latest checkpoint."
    }

    $state = $stateJson | ConvertFrom-Json

    Write-Host `
        "CER bad epochs: $($state.cer_bad_epochs)"

    if (
        [int]$state.cer_bad_epochs -ge 3
    ) {
        Write-Host ""
        Write-Host `
            "Stopping: no overall CER improvement for 3 epochs."

        break
    }

    $resumeCheckpoint = $latestCheckpoint
}

Write-Host ""
Write-Host "Fine-tuning finished."
Write-Host "Metrics: $metricsPath"
Write-Host "Checkpoints: $checkpointDirectory"
