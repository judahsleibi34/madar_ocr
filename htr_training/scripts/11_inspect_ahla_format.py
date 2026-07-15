from pathlib import Path

root = Path("data/raw/ahla")
output = Path("outputs/ahla_annotation_samples.txt")
output.parent.mkdir(parents=True, exist_ok=True)

report = []

readme_files = list(root.rglob("README*"))
yaml_files = list(root.rglob("*.yaml"))
label_files = list(root.rglob("labels/*.txt"))

for title, files in (
    ("README FILES", readme_files),
    ("YAML FILES", yaml_files),
    ("SAMPLE LABEL FILES", label_files[:10]),
):
    report.append("=" * 80)
    report.append(title)
    report.append("=" * 80)

    if not files:
        report.append("None found.")
        continue

    for path in files:
        report.append(f"\nPATH: {path}")

        try:
            text = path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )
            report.extend(text.splitlines()[:40])
        except OSError as error:
            report.append(f"ERROR: {error}")

text = "\n".join(report)
output.write_text(text, encoding="utf-8")

print(text)
print()
print(f"Saved report: {output.resolve()}")
