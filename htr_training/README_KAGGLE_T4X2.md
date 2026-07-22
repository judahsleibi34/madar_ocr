# Kaggle TrOCR T4 x2

## Files to commit to GitHub

- `scripts/31_train_trocr_kaggle_t4x2.py`
- `scripts/32_prepare_kaggle_ihwwr_upload.py`
- `requirements-kaggle.txt`

Do not commit the image ZIP, checkpoints, or model weights to GitHub.

## Build the private Kaggle dataset locally

From the project root:

```powershell
python .\scripts\32_prepare_kaggle_ihwwr_upload.py
```

This creates:

```text
kaggle_upload\ihwwr_trocr_data.zip
```

The default ZIP contains only `train_model.csv`, `val.csv`, and their images.
Keep `test.csv` local until final model selection. To include it deliberately:

```powershell
python .\scripts\32_prepare_kaggle_ihwwr_upload.py --include-test
```

Upload `ihwwr_trocr_data.zip` as one **private Kaggle Dataset**.

## Kaggle notebook setup

Enable **GPU T4 x2** and **Internet On**, then attach the private dataset.

```python
!nvidia-smi -L
```

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git /kaggle/working/project
%cd /kaggle/working/project
!pip install -q -r requirements-kaggle.txt
```

Launch two processes, one per T4:

```python
!torchrun --standalone --nproc_per_node=2 \
    scripts/31_train_trocr_kaggle_t4x2.py \
    --output /kaggle/working/trocr_ihwwr_t4x2 \
    --epochs 3 \
    --learning-rate 2e-5 \
    --train-batch-size 1 \
    --eval-batch-size 16 \
    --gradient-accumulation 8 \
    --eval-steps 1000 \
    --logging-steps 50 \
    --early-stopping-patience 3 \
    --resume auto
```

Global effective batch:

```text
1 image per GPU × 8 accumulation × 2 GPUs = 16
```

The script automatically discovers and extracts `ihwwr_trocr_data.zip` from
`/kaggle/input`. It saves checkpoints, `best_model`, metrics, and a final ZIP in
`/kaggle/working`.

Submit with **Save Version → Save & Run All** so the run continues remotely.
