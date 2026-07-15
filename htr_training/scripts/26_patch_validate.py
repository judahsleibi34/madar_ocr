from pathlib import Path

path = Path("scripts/16_train_crnn_dual_gpu.py")
content = path.read_text(encoding="utf-8")

old = '''        predictions = logits.argmax(dim=2)
        hypotheses = encoder.decode_batch(predictions)

        all_hypotheses.extend(hypotheses)
        all_references.extend(batch["texts"])
        all_languages.extend(batch["languages"])'''

new = '''        predictions = logits.argmax(dim=2)
        hypotheses = encoder.decode_batch(predictions)

        # decode_batch() returns visual-order text (matching how targets
        # were encoded), so references must be converted to the same
        # visual order before CER/WER comparison -- otherwise Arabic
        # references would still be in logical order and every Arabic
        # sample would score as ~100% wrong regardless of model quality.
        visual_references = [
            encoder.to_visual(text) for text in batch["texts"]
        ]

        all_hypotheses.extend(hypotheses)
        all_references.extend(visual_references)
        all_languages.extend(batch["languages"])'''

if old not in content:
    raise SystemExit("PATCH FAILED: old block not found -- file may already be modified or differ from expected.")

content = content.replace(old, new)
path.write_text(content, encoding="utf-8")
print("Patched successfully.")
