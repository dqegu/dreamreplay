import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_temporal_context_experiment import main as run_once, OUT_DIR

DEFAULT_SEEDS = [0, 7, 42, 123, 999]
KEYS = [
    'iid_clean_successor_mse', 'seq_clean_successor_mse',
    'iid_masked_successor_mse', 'seq_masked_successor_mse',
    'iid_masked_context_mse', 'seq_masked_context_mse',
    'iid_context_gain', 'seq_context_gain',
    'iid_rollout_mean_mse', 'seq_rollout_mean_mse',
    'iid_contiguity_gap', 'seq_contiguity_gap',
]


def aggregate(results):
    out = {}
    for k in KEYS:
        vals = np.array([r[k] for r in results], dtype=float)
        out[k] = {
            'values': vals.tolist(),
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max()),
        }
    return out


def save_summary(summary, seeds, out_dir):
    txt_path = os.path.join(out_dir, 'multirun_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('=' * 72 + '\n')
        f.write('Temporal-context multirun summary\n')
        f.write('=' * 72 + '\n')
        f.write(f'Seeds: {seeds}\n\n')
        focus_pairs = [
            ('clean successor', 'iid_clean_successor_mse', 'seq_clean_successor_mse', 'lower'),
            ('masked successor', 'iid_masked_successor_mse', 'seq_masked_successor_mse', 'lower'),
            ('masked + context', 'iid_masked_context_mse', 'seq_masked_context_mse', 'lower'),
            ('context gain', 'iid_context_gain', 'seq_context_gain', 'higher'),
            ('rollout mean', 'iid_rollout_mean_mse', 'seq_rollout_mean_mse', 'lower'),
            ('contiguity gap', 'iid_contiguity_gap', 'seq_contiguity_gap', 'higher'),
        ]
        for label, iid_key, seq_key, better in focus_pairs:
            iid = summary[iid_key]
            seq = summary[seq_key]
            f.write(f'{label}\n')
            f.write('-' * 40 + '\n')
            f.write(f"IID mean±std: {iid['mean']:.6f} ± {iid['std']:.6f}\n")
            f.write(f"SEQ mean±std: {seq['mean']:.6f} ± {seq['std']:.6f}\n")
            if better == 'lower':
                wins = sum(s < i for s, i in zip(seq['values'], iid['values']))
            else:
                wins = sum(s > i for s, i in zip(seq['values'], iid['values']))
            f.write(f'Seq wins: {wins}/{len(seeds)} runs\n\n')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(seeds))

    axes[0].bar(x - 0.18, summary['iid_masked_successor_mse']['values'], 0.36, label='IID', color='steelblue', alpha=0.8)
    axes[0].bar(x + 0.18, summary['seq_masked_successor_mse']['values'], 0.36, label='SEQ', color='coral', alpha=0.8)
    axes[0].set_title('Masked successor MSE')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in seeds], fontsize=8)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.18, summary['iid_masked_context_mse']['values'], 0.36, color='steelblue', alpha=0.8)
    axes[1].bar(x + 0.18, summary['seq_masked_context_mse']['values'], 0.36, color='coral', alpha=0.8)
    axes[1].set_title('Masked + context MSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in seeds], fontsize=8)

    axes[2].bar(x - 0.18, summary['iid_context_gain']['values'], 0.36, color='steelblue', alpha=0.8)
    axes[2].bar(x + 0.18, summary['seq_context_gain']['values'], 0.36, color='coral', alpha=0.8)
    axes[2].set_title('Context gain')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(s) for s in seeds], fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'multirun_summary.png'), dpi=160)
    plt.close(fig)


def main(seeds):
    out_dir = os.path.join(OUT_DIR, 'multirun')
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    for seed in seeds:
        run_dir = os.path.join(out_dir, f'run_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        print(f'\nRunning seed {seed}...')
        all_results.append(run_once(seed=seed, run_dir=run_dir))
    summary = aggregate(all_results)
    with open(os.path.join(out_dir, 'multirun_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    save_summary(summary, seeds, out_dir)
    print(f'\nSaved multirun summary to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args.seeds)
