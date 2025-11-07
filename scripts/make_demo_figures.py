"""Collect generated result PNGs into a single PDF for easy viewing.

Creates `results/demo_figures.pdf`.
"""
import os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, 'results')
OUT = os.path.join(RESULTS, 'demo_figures.pdf')

pngs = []
for name in os.listdir(RESULTS):
    if name.endswith('.png'):
        pngs.append(os.path.join(RESULTS, name))

if not pngs:
    print('No PNGs found in results/ to collate')
else:
    imgs = [Image.open(p).convert('RGB') for p in pngs]
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:])
    print(f'Wrote figures PDF -> {OUT}')
