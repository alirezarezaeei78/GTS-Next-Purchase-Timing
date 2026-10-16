"""Regenerate three-dataset manuscript diagnostics from bundled predictions.

These plots use the archived GTS ensemble, including the same-item Instacart
comparison. They do not mix in the fresh fixed/adaptive experiment or refit models.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = [('OnlineRetailII', 'Online Retail II'), ('TaFeng', 'Ta-Feng'),
            ('Instacart', 'Instacart')]
BLUE, GOLD = '#176B92', '#B66A22'


def generate(root=ROOT):
    root = Path(root)
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    records, grouped, histograms, tails = {}, [], [], []
    for dataset, title in DATASETS:
        same_item = dataset == 'Instacart'
        if same_item:
            frame = pd.read_parquet(root / 'data/instacart_pairs_paired_predictions.parquet')
            y, p = frame.actual_gap_h1.to_numpy(), frame.GTS_h1.to_numpy()
        else:
            frame = pd.read_csv(root / 'data' / f'{dataset}_paired_predictions.csv')
            y, p = frame.actual_days.to_numpy(), frame.dgt_full.to_numpy()
        assert len(y) == len(p) and np.isfinite(y).all() and np.isfinite(p).all()
        assert (y >= 0).all() and (p >= 0).all()
        groups = np.array_split(np.argsort(p, kind='stable'), 10)
        units = 'recorded days' if same_item else 'days'
        for index, rows in enumerate(groups, 1):
            grouped.append(dict(dataset=dataset, group=index, n=len(rows),
                                mean_observed=y[rows].mean(), mean_predicted=p[rows].mean(),
                                units=units))
        # Full observed range; every test outcome contributes once to the histogram.
        counts, edges = np.histogram(y, bins=30, range=(0, float(y.max())))
        assert int(counts.sum()) == len(y)
        for index, count in enumerate(counts):
            histograms.append(dict(dataset=dataset, bin_left=edges[index],
                                   bin_right=edges[index + 1], n=int(count),
                                   percent=100 * count / len(y), units=units))
        q = np.quantile(y, .9)
        selected, ae = y >= q, np.abs(p - y)
        tails.append(dict(dataset=dataset, test_q90=q, upper_group_n=int(selected.sum()),
                          upper_group_fraction=selected.mean(),
                          upper_group_error_share=ae[selected].sum() / ae.sum(),
                          upper_group_MAE=ae[selected].mean(), remaining_MAE=ae[~selected].mean()))
        records[dataset] = dict(y=y, title=title, counts=counts, edges=edges)

    points = pd.DataFrame(grouped)
    bins = pd.DataFrame(histograms)
    points.to_csv(root / 'data/group_calibration_points.csv', index=False)
    bins.to_csv(root / 'data/gap_distribution_bins.csv', index=False)
    pd.DataFrame(tails).to_csv(root / 'data/tail_diagnostics.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})

    def save(fig, name):
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(root / 'figures' / f'{name}.{ext}', bbox_inches='tight', dpi=260)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), layout='constrained')
    for letter, ax, (dataset, title) in zip('ABC', axes, DATASETS):
        rec = records[dataset]
        y, edges, counts = rec['y'], rec['edges'], rec['counts']
        ax.bar(edges[:-1], 100 * counts / len(y), width=np.diff(edges), align='edge',
               color=BLUE, edgecolor='white', linewidth=.4)
        ax.axvline(np.median(y), color='#222222', linestyle='--', linewidth=1.3)
        ax.text(.97, .95, f'Median: {np.median(y):g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10.5,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2})
        ax.set_title(f'{letter}  {title}\n{len(y):,} test origins', pad=9)
        ax.set(xlabel=('Same-item gap\n(recorded days)' if dataset == 'Instacart'
                       else 'Next purchase-day gap\n(days)'), ylabel='Test outcomes (%)',
               xlim=(0, float(y.max())))
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    save(fig, 'gap_distributions')

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), layout='constrained')
    for letter, ax, (dataset, title) in zip('ABC', axes, DATASETS):
        d = points[points.dataset == dataset]
        ax.plot(d.group, d.mean_observed, 'o-', color=GOLD, markersize=4,
                label='Mean observed')
        ax.plot(d.group, d.mean_predicted, 's--', color=BLUE, markersize=4,
                label='Mean predicted')
        ax.set_title(f'{letter}  {title}', pad=9)
        ax.set(xlabel='Group ordered by prediction',
               ylabel=('Mean gap (recorded days)' if dataset == 'Instacart'
                       else 'Mean gap (days)'), xticks=[1, 3, 5, 7, 10], xlim=(.7, 10.3))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=10, frameon=False, loc='upper left')
    save(fig, 'group_calibration')
    return tails


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    generate(args.root)
    print('Saved complete distributions, grouped means, and tail diagnostics for all three public datasets.')
