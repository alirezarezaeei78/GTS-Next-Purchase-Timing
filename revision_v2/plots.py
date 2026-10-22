"""Publication figures from completed revision results only; no model fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'OnlineRetailII': 'Online Retail II', 'TaFeng': 'Ta-Feng', 'Instacart': 'Instacart'}
LABELS = {'selected': 'GTS: selected inputs', 'fixed': 'GTS: fixed inputs', 'encoder': 'Encoder alone',
          'boosting': 'Boosting', 'lstm': 'LSTM', 'gru': 'GRU', 'window_mean': 'Window mean',
          'window_median': 'Window median', 'last_gap': 'Last gap', 'deepar': 'DeepAR adaptation',
          'weibull': 'Neural Weibull AFT', 'mlp_mae': 'Small NN: MAE', 'mlp_mse': 'Small NN: MSE',
          'tuned_window_mean': 'Tuned window mean', 'tuned_window_median': 'Tuned window median',
          'training_median': 'Training median'}


def generate(root):
    assert (root / 'numerical_sections_complete.json').exists(), 'All numerical sections must finish before plotting.'
    assert (root / 'benchmark_complete.json').exists(), 'Saved-model inference verification must finish before plotting.'
    out = root / 'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    def save(fig, name):
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', dpi=250)
        plt.close(fig)
    m = pd.read_csv(root / 'metrics.csv')
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout='constrained')
    for i, (dataset, title) in enumerate(NAMES.items()):
        d = m[(m.dataset == dataset) & m.model.isin(LABELS)].set_index('model')
        order = d.sort_values('MAE', ascending=False).index
        for j, metric in enumerate(['MAE', 'RMSE']):
            ax = axes[i, j]
            ax.barh(np.arange(len(order)), d.loc[order, metric], color=['#176B92' if n == 'selected' else '#82929B' for n in order])
            ax.set(yticks=np.arange(len(order)), yticklabels=[LABELS[n] for n in order],
                   xlabel=f'{metric} ({"recorded days" if dataset == "Instacart" else "days"})', title=title)
            ax.set_xlim(left=0); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    save(fig, 'revision_model_comparison')
    grouped = pd.read_csv(root / 'grouped_predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout='constrained')
    for ax, (dataset, title) in zip(axes, NAMES.items()):
        d = grouped[grouped.dataset == dataset]
        ax.plot(d.group, d.mean_observed, 'o-', label='Observed mean', color='#B66A22', markersize=4)
        ax.plot(d.group, d.median_observed, '^:', label='Observed median', color='#555555', markersize=4)
        ax.plot(d.group, d.mean_prediction, 's--', label='Mean prediction', color='#176B92', markersize=4)
        ax.set(title=title, xlabel='Group ordered by prediction',
               ylabel='Gap (recorded days)' if dataset == 'Instacart' else 'Gap (days)', xticks=[1, 3, 5, 7, 10])
        ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8, frameon=False)
    save(fig, 'revision_grouped_predictions')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout='constrained')
    for ax, (dataset, title) in zip(axes, NAMES.items()):
        d = pd.read_parquet(root / f'{dataset}_test_predictions.parquet')
        weights = np.full(len(d), 100 / len(d))
        ax.hist(d.duration, bins=30, weights=weights, color='#176B92', edgecolor='white', linewidth=.4)
        ax.axvline(d.duration.median(), color='black', linestyle='--', linewidth=1)
        ax.set(title=f'{title}\n{len(d):,} complete test outcomes', ylabel='Outcomes (%)',
               xlabel='Same-item gap\n(recorded days)' if dataset == 'Instacart' else 'Purchase-day gap\n(days)')
        ax.set_xlim(left=0); ax.grid(axis='y', alpha=.15)
    save(fig, 'revision_gap_distributions')
    w = pd.read_csv(root / 'combiner_weights.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout='constrained')
    colors = ['#176B92', '#B66A22', '#74828B', '#635C9C', '#94ADAA']
    for ax, (dataset, title) in zip(axes, NAMES.items()):
        d = w[(w.dataset == dataset) & (w.arm == 'selected')].pivot(index='seed', columns='member', values='weight')
        columns = ['encoder', 'boosting', 'lstm', 'gru', 'window_mean']
        bottom = np.zeros(len(d))
        for column, color in zip(columns, colors):
            ax.bar(np.arange(len(d)), d[column], bottom=bottom, label=column, color=color)
            bottom += d[column].to_numpy()
        ax.set(title=title, xticks=np.arange(len(d)), xticklabels=d.index.astype(str), ylabel='Simplex weight', ylim=(0, 1))
        ax.tick_params(axis='x', rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=8, frameon=False)
    save(fig, 'revision_combiner_weights')
    a = pd.read_csv(root / 'ablation_metrics.csv'); seed = pd.read_csv(root / 'seed_metrics.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout='constrained')
    for ax, (dataset, title) in zip(axes, NAMES.items()):
        d = a[(a.dataset == dataset) & (a.scope == 'full_system') & a.variant.str.startswith('graph_P')].copy()
        d['P'] = d.variant.str.extract(r'P(\d+)').astype(int)
        d['d'] = d.variant.str.extract(r'_d(\d+)').astype(int)
        baseline = float(seed[(seed.dataset == dataset) & (seed.seed == 42) & (seed.model == 'selected')].MAE.iloc[0])
        table = d.pivot(index='P', columns='d', values='MAE').sort_index()
        delta = table.to_numpy() - baseline
        limit = max(.001, float(np.abs(delta).max()))
        image = ax.imshow(delta, cmap='PuOr_r', vmin=-limit, vmax=limit, aspect='auto')
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f'{delta[i,j]:+.3f}', ha='center', va='center', fontsize=9)
        ax.set(title=title, xticks=np.arange(len(table.columns)), xticklabels=table.columns,
               yticks=np.arange(len(table.index)), yticklabels=table.index, xlabel='Graph dimensions', ylabel='Represented products')
        fig.colorbar(image, ax=ax, shrink=.8, label='MAE change: seed 42')
    save(fig, 'revision_graph_sensitivity')
    h = pd.read_csv(root / 'horizon_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout='constrained')
    for ax, metric in zip(axes, ['MAE', 'RMSE']):
        for model, marker in [('GTS', 'o'), ('transformer_direct', 's'), ('transformer_recursive', '^'), ('window_median', 'D')]:
            d = h[h.model == model].sort_values('horizon')
            ax.plot(d.horizon, d[metric], marker=marker, label=model.replace('_', ' '))
        ax.set(xlabel='Forecast horizon', ylabel=f'{metric} (days)', xticks=range(1, 6)); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, 'revision_five_horizons')
    print('Saved six result figures, with Instacart included in every applicable three-dataset comparison.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    generate(parser.parse_args().results)
