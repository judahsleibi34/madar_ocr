import importlib.util
from pathlib import Path

from torch.utils.data import DataLoader


MODULE_PATH = Path("scripts/08_crnn_dataset.py")

spec = importlib.util.spec_from_file_location(
    "crnn_dataset",
    MODULE_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module: {MODULE_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CRNNDataset = module.CRNNDataset
ctc_collate_fn = module.ctc_collate_fn


def test_manifest(manifest_path: str) -> None:
    dataset = CRNNDataset(
        manifest_path=manifest_path,
        vocab_path="data/processed/vocab.json",
        image_height=64,
        max_image_width=1024,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
    )

    batch = next(iter(loader))

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Images shape: {batch['images'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"Target lengths: {batch['target_lengths']}")
    print(f"Image widths: {batch['image_widths']}")
    print(f"Languages: {batch['languages']}")
    print(f"Datasets: {batch['datasets']}")
    print(f"First text: {batch['texts'][0]}")


def main() -> None:
    test_manifest("data/processed/iam/train.csv")
    test_manifest("data/processed/arabic/train.csv")


if __name__ == "__main__":
    main()
