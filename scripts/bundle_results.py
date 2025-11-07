"""Create a zip bundle of key results and docs for committee submission.

Creates `results/demo_bundle.zip` containing `results/` JSONs/PNGs and `docs/README_results.md`.
"""
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "demo_bundle.zip")

files_to_include = []
for name in os.listdir(RESULTS):
    if name.endswith('.json') or name.endswith('.png') or name.endswith('.npz') or name.endswith('.pth'):
        files_to_include.append(os.path.join(RESULTS, name))

# include README_results
rd = os.path.join(ROOT, 'docs', 'README_results.md')
if os.path.exists(rd):
    files_to_include.append(rd)

with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in files_to_include:
        arcname = os.path.relpath(p, ROOT)
        z.write(p, arcname)

print(f"Created bundle -> {OUT}")
