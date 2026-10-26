"""Coverage vs correctness chart for the README.

Numbers: MoveHarness results/baselines.md (github.com/dhruvinci/moveharness), "Pipeline A" = Experiment 3,
"Pipeline B" = Experiment 4, scored by the MoveBench scorer against one hand-labelled 16-minute match
(strict exact-class matching; a lower bound).
"""
import matplotlib.pyplot as plt
import numpy as np

rows = ['Exp 3: CV constraints\n+ Gemini (58 segments)', 'Exp 4: skeleton-flesh\n(25 segments)']
series = [('timeline covered', [1.00, 1.00], '#9a9993'),
          ('matched-span overlap (IoU)', [0.463, 0.723], '#2a78d6'),
          ('position macro-F1 (strict)', [0.099, 0.136], '#eb6834')]
fig, ax = plt.subplots(figsize=(8.4, 3.6), facecolor='#fcfcfb')
ax.set_facecolor('#fcfcfb')
y = np.arange(len(rows))
h = 0.25
for i, (name, vals, col) in enumerate(series):
    bars = ax.barh(y + (1 - i) * h, vals, h * 0.9, color=col, label=name)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f'{v:.2f}', va='center', fontsize=9, color='#333')
ax.set_yticks(y, rows)
ax.set_xlim(0, 1.12)
ax.set_xlabel('score (0-1)')
ax.set_title('Full coverage is not correctness: one hand-labelled match, scored later by MoveBench', fontsize=10.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(loc='lower right', fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig('docs/assets/coverage_vs_correctness.png', dpi=130)
