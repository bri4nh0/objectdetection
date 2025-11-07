"""Create a simple 5-slide PDF summarizing claims and key figures.

Produces `results/demo_slides.pdf`.
"""
import os
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, 'results', 'demo_slides.pdf')

def slide(text, subtitle=None, fig=None):
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.title(text, fontsize=20)
    if subtitle:
        plt.text(0.1, 0.6, subtitle, fontsize=14)
    if fig is not None:
        plt.imshow(fig)
    return plt.gcf()

slides = []
slides.append(slide('TRD-UQ: TinyDeepEnsemble', 'Low-latency ensemble via shared backbone'))
slides.append(slide('Contributions', '- TinyDeepEnsemble\n- Low-rank heteroscedastic head\n- Runtime-adaptive prototype'))
slides.append(slide('Demo results', 'See bundled figures in demo_figures.pdf'))
slides.append(slide('Rank sweep', 'Low-rank hetero head improves calibration'))
slides.append(slide('Reproducibility', 'All artifacts bundled: results/demo_bundle.zip'))

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(OUT) as pdf:
    for fig in slides:
        pdf.savefig(fig)
        plt.close(fig)

print(f'Wrote slides -> {OUT}')
