import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_module(module_name: str, file_path: str):
    path = Path(file_path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


dataset_module = load_module(
    "crnn_dataset",
    "scripts/08_crnn_dataset.py",
)

model_module = load_module(
    "crnn_model",
    "scripts/10_crnn_model.py",
)

CRNNDataset = dataset_module.CRNNDataset
ctc_collate_fn = dataset_module.ctc_collate_fn
CRNN = model_module.CRNN


def main() -> None:
    vocab_path = Path("data/processed/vocab.json")

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}"
        )

    with vocab_path.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    num_classes = int(vocabulary["num_classes"])

    dataset = CRNNDataset(
        manifest_path="data/processed/iam/train.csv",
        vocab_path=vocab_path,
        image_height=64,
        max_image_width=1600,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
    )

    batch = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CRNN(
        num_classes=num_classes,
        hidden_size=256,
        dropout=0.1,
    ).to(device)

    images = batch["images"].to(device)

    with torch.no_grad():
        logits = model(images)

    input_lengths = model.calculate_input_lengths(
        batch["image_widths"]
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Number of classes: {num_classes}")
    print(f"Images shape: {images.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Input lengths: {input_lengths}")
    print(f"Target lengths: {batch['target_lengths']}")

    if logits.ndim != 3:
        raise RuntimeError(
            f"Expected logits with 3 dimensions, got {logits.shape}"
        )

    if logits.shape[1] != images.shape[0]:
        raise RuntimeError(
            "The logits batch dimension does not match the image batch."
        )

    if logits.shape[2] != num_classes:
        raise RuntimeError(
            "The logits class dimension does not match the vocabulary."
        )

    if torch.any(
        batch["target_lengths"] > input_lengths
    ):
        raise RuntimeError(
            "At least one target is longer than the available "
            "CTC time sequence."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Parameters: {parameter_count:,}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )

    print("CRNN model test passed.")


if __name__ == "__main__":
    main()
