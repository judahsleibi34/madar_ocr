param(
    [int]$TargetEpochs = 8,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 2
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$runName = "bilingual_v4_ar70_en30_ft"

$trainManifest = `
    ".\data\processed\master_ctc\train_finetune_ar70_en30.csv"

$validationManifest = `
    ".\data\processed\master_ctc\val_all.csv"

$vocab = `
    ".\data\processed\master_safe\vocab.json"

$checkpointDirectory = `
    Join-Path ".\outputs\checkpoints" $runName

$latestCheckpoint = `
    Join-Path $checkpointDirectory "latest.pt"

$required = @(
    ".\scripts\23_train_crnn_resumable.py",
    $trainManifest,
    $validationManifest,
    $vocab,
    $latestCheckpoint
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required path: $path"
    }
}

function Read-CheckpointState {
    param(
        [string]$CheckpointPath
    )

    $stateJson = & python -c `
        "import json,sys,torch; p=sys.argv[1]; c=torch.load(p,map_location='cpu',weights_only=False); print(json.dumps({'epoch':int(c.get('epoch',0)),'cer_bad_epochs':int(c.get('cer_bad_epochs',0))}))" `
        $CheckpointPath

    if ($LASTEXITCODE -ne 0) {
        throw "Could not read checkpoint: $CheckpointPath"
    }

    return $stateJson | ConvertFrom-Json
}

$state = Read-CheckpointState `
    -CheckpointPath $latestCheckpoint

$currentEpoch = [int]$state.epoch

Write-Host ""
Write-Host "Existing checkpoint epoch: $currentEpoch"
Write-Host "Target epoch: $TargetEpochs"

if ($currentEpoch -ge $TargetEpochs) {
    Write-Host "The requested epochs are already complete."
    exit 0
}

$currentEpochCopy = Join-Path `
    $checkpointDirectory `
    ("epoch_{0:D2}.pt" -f $currentEpoch)

Copy-Item `
    $latestCheckpoint `
    $currentEpochCopy `
    -Force

Write-Host "Preserved: $currentEpochCopy"

for (
    $epoch = $currentEpoch + 1;
    $epoch -le $TargetEpochs;
    $epoch++
) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Fine-tuning epoch $epoch of $TargetEpochs"
    Write-Host "========================================"

    $trainingArguments = @(
        ".\scripts\23_train_crnn_resumable.py",
        "--train-manifest",
        $trainManifest,
        "--validation-manifest",
        $validationManifest,
        "--run-name",
        $runName,
        "--vocab",
        $vocab,
        "--resume",
        $latestCheckpoint,
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

    & python @trainingArguments

    # Save this immediately. Some local setups report a nonzero
    # process code even though the trainer completed and saved.
    $trainingExitCode = $LASTEXITCODE

    if (-not (Test-Path $latestCheckpoint)) {
        throw "Training did not create latest.pt for epoch $epoch."
    }

    $newState = Read-CheckpointState `
        -CheckpointPath $latestCheckpoint

    $savedEpoch = [int]$newState.epoch

    if ($savedEpoch -ne $epoch) {
        throw @"
Training did not finish epoch $epoch.

Checkpoint still reports epoch $savedEpoch.
Python exit code: $trainingExitCode
"@
    }

    if ($trainingExitCode -ne 0) {
        Write-Warning @"
Python returned exit code $trainingExitCode, but epoch $epoch
completed and the checkpoint was verified. Continuing safely.
"@
    }

    $epochCheckpoint = Join-Path `
        $checkpointDirectory `
        ("epoch_{0:D2}.pt" -f $epoch)

    Copy-Item `
        $latestCheckpoint `
        $epochCheckpoint `
        -Force

    Write-Host "Verified and preserved: $epochCheckpoint"

    $badEpochs = [int]$newState.cer_bad_epochs

    Write-Host "CER bad epochs: $badEpochs"

    if ($badEpochs -ge 3) {
        Write-Host ""
        Write-Host "Early stopping after 3 epochs without CER improvement."
        break
    }
}

Write-Host ""
Write-Host "Fine-tuning continuation finished."
Write-Host "Checkpoints: $checkpointDirectory"
Write-Host "Metrics: .\outputs\metrics\$runName.csv"
