import os
import json
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import run_seq_replay as base_exp


BASE_DIR = base_exp.BASE_DIR
ART_DIR = base_exp.ART_DIR
OUT_DIR = os.path.join(BASE_DIR, 'outputs_temporal_context')
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def student_paths(seed: int):
    return {
        'iid_enc': os.path.join(ART_DIR, f'student_iid_encoder_s{seed}.keras'),
        'iid_dec': os.path.join(ART_DIR, f'student_iid_decoder_s{seed}.keras'),
        'seq_enc': os.path.join(ART_DIR, f'student_seq_encoder_s{seed}.keras'),
        'seq_dec': os.path.join(ART_DIR, f'student_seq_decoder_s{seed}.keras'),
        'seq_trans': os.path.join(ART_DIR, f'student_seq_transition_s{seed}.keras'),
    }


def ensure_models_exist(seed: int):
    paths = student_paths(seed)
    need_bootstrap = any(not os.path.exists(p) for p in paths.values())
    need_bootstrap = need_bootstrap or (not os.path.exists(base_exp.ENC_PATH)) or (not os.path.exists(base_exp.DEC_PATH))
    need_bootstrap = need_bootstrap or (not os.path.exists(base_exp.K_PATH)) or (not os.path.exists(base_exp.V_PATH))
    if need_bootstrap:
        print('[bootstrap] Missing teacher/student artifacts. Running base experiment first...')
        bootstrap_dir = os.path.join(OUT_DIR, f'bootstrap_seed_{seed}')
        os.makedirs(bootstrap_dir, exist_ok=True)
        base_exp.main(seed=seed, run_dir=bootstrap_dir)
    return paths


def load_models(seed: int):
    paths = ensure_models_exist(seed)
    teacher_enc = keras.models.load_model(base_exp.ENC_PATH, compile=False)
    teacher_dec = keras.models.load_model(base_exp.DEC_PATH, compile=False)
    student_iid_enc = keras.models.load_model(paths['iid_enc'], compile=False)
    student_iid_dec = keras.models.load_model(paths['iid_dec'], compile=False)
    student_seq_enc = keras.models.load_model(paths['seq_enc'], compile=False)
    student_seq_dec = keras.models.load_model(paths['seq_dec'], compile=False)
    student_seq_trans = keras.models.load_model(paths['seq_trans'], compile=False)
    return teacher_enc, teacher_dec, student_iid_enc, student_iid_dec, student_seq_enc, student_seq_dec, student_seq_trans


def load_full_shapes3d(n_samples: int):
    imgs, labels = base_exp.load_shapes3d_with_labels(n_samples)
    split = int(0.8 * n_samples)
    train_imgs, train_labels = imgs[:split], labels[:split]
    test_imgs, test_labels = imgs[split:], labels[split:]
    return train_imgs, train_labels, test_imgs, test_labels


def build_sequences(labels):
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        gkey = tuple(int(lab[k]) for k in base_exp.GROUP_LABELS)
        groups[gkey].append((int(lab[base_exp.SEQ_LABEL]), i))
    sequences = []
    for _, seq in groups.items():
        seq_sorted = [idx for _, idx in sorted(seq, key=lambda x: x[0])]
        if len(seq_sorted) >= 5:
            sequences.append(seq_sorted)
    return sequences


def sample_triplets(imgs, sequences, n_triplets: int, rng: np.random.Generator):
    prevs, curs, nexts = [], [], []
    for _ in range(n_triplets):
        seq = sequences[int(rng.integers(len(sequences)))]
        t = int(rng.integers(1, len(seq) - 1))
        prevs.append(imgs[seq[t - 1]])
        curs.append(imgs[seq[t]])
        nexts.append(imgs[seq[t + 1]])
    return np.stack(prevs), np.stack(curs), np.stack(nexts)


def sample_pairs(imgs, sequences, n_pairs: int, rng: np.random.Generator):
    xs, ys = [], []
    for _ in range(n_pairs):
        seq = sequences[int(rng.integers(len(sequences)))]
        t = int(rng.integers(0, len(seq) - 1))
        xs.append(imgs[seq[t]])
        ys.append(imgs[seq[t + 1]])
    return np.stack(xs), np.stack(ys)


def sample_rollout_windows(imgs, sequences, horizon: int, n_windows: int, rng: np.random.Generator):
    starts = []
    futures = []
    valid_sequences = [seq for seq in sequences if len(seq) >= horizon + 1]
    for _ in range(n_windows):
        seq = valid_sequences[int(rng.integers(len(valid_sequences)))]
        t = int(rng.integers(0, len(seq) - horizon))
        starts.append(imgs[seq[t]])
        futures.append([imgs[seq[t + h]] for h in range(1, horizon + 1)])
    return np.stack(starts), np.stack(futures)


def center_mask(images, frac: float = 0.45):
    out = images.copy()
    h, w = out.shape[1:3]
    mh = int(h * frac)
    mw = int(w * frac)
    y0 = (h - mh) // 2
    x0 = (w - mw) // 2
    out[:, y0:y0 + mh, x0:x0 + mw, :] = 0.0
    return out


def gaussian_noise(images, sigma: float = 0.20):
    noisy = images + np.random.normal(0.0, sigma, size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def encode_mean(encoder, images, batch_size: int = 64):
    return encoder.predict(images, batch_size=batch_size, verbose=0)[0]


def fit_probe(x_train, y_train, alpha: float = 1.0):
    probe = Ridge(alpha=alpha)
    probe.fit(x_train, y_train)
    return probe


def cosine_similarity_rows(a, b, eps: float = 1e-8):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(an * bn, axis=1)


def eval_clean_successor(encoder, train_x, train_y, test_x, test_y):
    z_x_train = encode_mean(encoder, train_x)
    z_y_train = encode_mean(encoder, train_y)
    z_x_test = encode_mean(encoder, test_x)
    z_y_test = encode_mean(encoder, test_y)
    probe = fit_probe(z_x_train, z_y_train)
    pred = probe.predict(z_x_test)
    return float(mean_squared_error(z_y_test, pred))


def eval_corrupted_successor(encoder, train_x, train_y, test_x, test_y, corruption='mask'):
    z_x_train = encode_mean(encoder, train_x)
    z_y_train = encode_mean(encoder, train_y)
    probe = fit_probe(z_x_train, z_y_train)

    if corruption == 'mask':
        test_x_corr = center_mask(test_x)
    elif corruption == 'noise':
        test_x_corr = gaussian_noise(test_x)
    else:
        raise ValueError(f'Unknown corruption: {corruption}')

    z_x_test = encode_mean(encoder, test_x_corr)
    z_y_test = encode_mean(encoder, test_y)
    pred = probe.predict(z_x_test)
    return float(mean_squared_error(z_y_test, pred)), test_x_corr


def eval_context_gain(encoder, prev_train, cur_train, next_train, prev_test, cur_test, next_test, corruption='mask'):
    if corruption == 'mask':
        cur_train_corr = center_mask(cur_train)
        cur_test_corr = center_mask(cur_test)
    elif corruption == 'noise':
        cur_train_corr = gaussian_noise(cur_train)
        cur_test_corr = gaussian_noise(cur_test)
    else:
        raise ValueError(f'Unknown corruption: {corruption}')

    z_prev_train = encode_mean(encoder, prev_train)
    z_cur_train_corr = encode_mean(encoder, cur_train_corr)
    z_next_train = encode_mean(encoder, next_train)

    z_prev_test = encode_mean(encoder, prev_test)
    z_cur_test_corr = encode_mean(encoder, cur_test_corr)
    z_next_test = encode_mean(encoder, next_test)

    probe_no_ctx = fit_probe(z_cur_train_corr, z_next_train)
    pred_no_ctx = probe_no_ctx.predict(z_cur_test_corr)
    mse_no_ctx = float(mean_squared_error(z_next_test, pred_no_ctx))

    x_train_ctx = np.concatenate([z_prev_train, z_cur_train_corr], axis=1)
    x_test_ctx = np.concatenate([z_prev_test, z_cur_test_corr], axis=1)
    probe_ctx = fit_probe(x_train_ctx, z_next_train)
    pred_ctx = probe_ctx.predict(x_test_ctx)
    mse_ctx = float(mean_squared_error(z_next_test, pred_ctx))

    gain = mse_no_ctx - mse_ctx
    return mse_no_ctx, mse_ctx, float(gain), cur_test_corr


def eval_rollout_mse(encoder, start_train, next_train, start_test, future_test, horizon: int):
    z_x_train = encode_mean(encoder, start_train)
    z_y_train = encode_mean(encoder, next_train)
    probe = fit_probe(z_x_train, z_y_train)

    z_start = encode_mean(encoder, start_test)
    z_true_future = np.stack([
        encode_mean(encoder, future_test[:, h]) for h in range(horizon)
    ], axis=1)

    z_curr = z_start.copy()
    step_mse = []
    for h in range(horizon):
        z_curr = probe.predict(z_curr)
        mse_h = float(mean_squared_error(z_true_future[:, h, :], z_curr))
        step_mse.append(mse_h)
    return step_mse, float(np.mean(step_mse))


def eval_temporal_contiguity(encoder, imgs, sequences, max_sequences: int = 100):
    adj_sims = []
    far_sims = []
    for seq in sequences[:max_sequences]:
        z = encode_mean(encoder, imgs[seq])
        if len(z) < 5:
            continue
        for i in range(len(z) - 1):
            adj_sims.append(float(cosine_similarity_rows(z[i:i+1], z[i+1:i+2])[0]))
        for i in range(len(z) - 3):
            far_sims.append(float(cosine_similarity_rows(z[i:i+1], z[i+3:i+4])[0]))
    adj_mean = float(np.mean(adj_sims)) if adj_sims else float('nan')
    far_mean = float(np.mean(far_sims)) if far_sims else float('nan')
    return adj_mean, far_mean, float(adj_mean - far_mean)


def save_metric_plot(results, out_path):
    metric_names = [
        'Clean successor MSE',
        'Masked successor MSE',
        'Masked+context MSE',
        'Context gain',
        '4-step rollout MSE',
        'Contiguity gap',
    ]
    iid_vals = [
        results['iid_clean_successor_mse'],
        results['iid_masked_successor_mse'],
        results['iid_masked_context_mse'],
        results['iid_context_gain'],
        results['iid_rollout_mean_mse'],
        results['iid_contiguity_gap'],
    ]
    seq_vals = [
        results['seq_clean_successor_mse'],
        results['seq_masked_successor_mse'],
        results['seq_masked_context_mse'],
        results['seq_context_gain'],
        results['seq_rollout_mean_mse'],
        results['seq_contiguity_gap'],
    ]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, iid_vals, width, label='IID', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, seq_vals, width, label='Sequential', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=25, ha='right')
    ax.set_title('Temporal-context evaluation: IID vs sequential replay')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_example_grid(prev, cur_corr, nxt, out_path, n: int = 8, title: str = ''):
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    labels = ['Previous frame', 'Current frame (corrupted)', 'True next frame']
    for i in range(n):
        axes[0, i].imshow(np.clip(prev[i], 0, 1))
        axes[1, i].imshow(np.clip(cur_corr[i], 0, 1))
        axes[2, i].imshow(np.clip(nxt[i], 0, 1))
        for r in range(3):
            axes[r, i].axis('off')
    for r, lab in enumerate(labels):
        axes[r, 0].set_ylabel(lab, fontsize=8, rotation=90, labelpad=40)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(seed: int = 42,
         n_samples: int = 2000,
         n_pair_train: int = 1500,
         n_pair_test: int = 400,
         n_triplet_train: int = 1500,
         n_triplet_test: int = 400,
         rollout_horizon: int = 4,
         n_rollout_train: int = 1500,
         n_rollout_test: int = 250,
         corruption: str = 'mask',
         run_dir: str = None):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = run_dir if run_dir else os.path.join(OUT_DIR, f'run_seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(f'Temporal-context experiment [seed={seed}]')
    print('=' * 70)

    teacher_enc, teacher_dec, iid_enc, iid_dec, seq_enc, seq_dec, seq_trans = load_models(seed)

    print('[1/4] Loading Shapes3D and building held-out sequences...')
    train_imgs, train_labels, test_imgs, test_labels = load_full_shapes3d(n_samples)
    train_sequences = build_sequences(train_labels)
    test_sequences = build_sequences(test_labels)
    print(f'  train sequences: {len(train_sequences)} | test sequences: {len(test_sequences)}')

    print('[2/4] Sampling train/test pairs and triplets...')
    pair_train_x, pair_train_y = sample_pairs(train_imgs, train_sequences, n_pair_train, rng)
    pair_test_x, pair_test_y = sample_pairs(test_imgs, test_sequences, n_pair_test, rng)

    trip_train_prev, trip_train_cur, trip_train_next = sample_triplets(train_imgs, train_sequences, n_triplet_train, rng)
    trip_test_prev, trip_test_cur, trip_test_next = sample_triplets(test_imgs, test_sequences, n_triplet_test, rng)

    roll_train_x, roll_train_y = sample_pairs(train_imgs, train_sequences, n_rollout_train, rng)
    roll_test_start, roll_test_future = sample_rollout_windows(test_imgs, test_sequences, rollout_horizon, n_rollout_test, rng)

    print('[3/4] Evaluating IID and sequential students...')
    results = {}
    for name, enc in [('iid', iid_enc), ('seq', seq_enc)]:
        clean_mse = eval_clean_successor(enc, pair_train_x, pair_train_y, pair_test_x, pair_test_y)
        masked_mse, _ = eval_corrupted_successor(enc, pair_train_x, pair_train_y, pair_test_x, pair_test_y, corruption=corruption)
        noctx_mse, ctx_mse, gain, cur_test_corr = eval_context_gain(
            enc,
            trip_train_prev, trip_train_cur, trip_train_next,
            trip_test_prev, trip_test_cur, trip_test_next,
            corruption=corruption,
        )
        rollout_steps, rollout_mean = eval_rollout_mse(
            enc, roll_train_x, roll_train_y, roll_test_start, roll_test_future, rollout_horizon
        )
        adj_mean, far_mean, gap = eval_temporal_contiguity(enc, test_imgs, test_sequences)

        results[f'{name}_clean_successor_mse'] = clean_mse
        results[f'{name}_masked_successor_mse'] = masked_mse
        results[f'{name}_masked_no_context_mse'] = noctx_mse
        results[f'{name}_masked_context_mse'] = ctx_mse
        results[f'{name}_context_gain'] = gain
        results[f'{name}_rollout_mean_mse'] = rollout_mean
        results[f'{name}_contiguity_adj_mean'] = adj_mean
        results[f'{name}_contiguity_far_mean'] = far_mean
        results[f'{name}_contiguity_gap'] = gap
        for i, v in enumerate(rollout_steps, start=1):
            results[f'{name}_rollout_step_{i}_mse'] = v

    # Optional direct sequential transition metric, not used as comparison headline.
    z_seq_test = encode_mean(seq_enc, pair_test_x)
    z_seq_true_next = encode_mean(seq_enc, pair_test_y)
    z_seq_pred_next = seq_trans.predict(z_seq_test, verbose=0)
    results['seq_direct_transition_mse'] = float(mean_squared_error(z_seq_true_next, z_seq_pred_next))

    print('[4/4] Saving plots and metrics...')
    _, example_corr = eval_corrupted_successor(seq_enc, pair_train_x, pair_train_y, pair_test_x[:8], pair_test_y[:8], corruption=corruption)
    save_example_grid(
        trip_test_prev[:8],
        center_mask(trip_test_cur[:8]) if corruption == 'mask' else gaussian_noise(trip_test_cur[:8]),
        trip_test_next[:8],
        os.path.join(out_dir, 'example_temporal_context_grid.png'),
        title='Temporal-context task: previous | current corrupted | next'
    )
    save_metric_plot(results, os.path.join(out_dir, 'temporal_context_metrics.png'))

    metrics_json = os.path.join(out_dir, 'metrics.json')
    with open(metrics_json, 'w') as f:
        json.dump(results, f, indent=2)

    summary_txt = os.path.join(out_dir, 'summary.txt')
    with open(summary_txt, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write(f'Temporal-context experiment summary [seed={seed}]\n')
        f.write('=' * 70 + '\n\n')
        f.write('Headline question: does sequential replay help under partial/ambiguous cues,\n')
        f.write('when the current frame alone is degraded and temporal context matters?\n\n')
        for prefix, label in [('iid', 'IID'), ('seq', 'Sequential')]:
            f.write(f'{label}\n')
            f.write('-' * 30 + '\n')
            f.write(f"Clean successor MSE:      {results[f'{prefix}_clean_successor_mse']:.6f}\n")
            f.write(f"Masked successor MSE:     {results[f'{prefix}_masked_successor_mse']:.6f}\n")
            f.write(f"Masked no-context MSE:    {results[f'{prefix}_masked_no_context_mse']:.6f}\n")
            f.write(f"Masked + context MSE:     {results[f'{prefix}_masked_context_mse']:.6f}\n")
            f.write(f"Context gain:             {results[f'{prefix}_context_gain']:.6f}\n")
            f.write(f"{rollout_horizon}-step rollout mean MSE: {results[f'{prefix}_rollout_mean_mse']:.6f}\n")
            f.write(f"Temporal contiguity gap:  {results[f'{prefix}_contiguity_gap']:.6f}\n\n")
        f.write('Interpretation guide\n')
        f.write('-' * 30 + '\n')
        f.write('* Lower is better for all MSE metrics.\n')
        f.write('* Higher is better for context gain and contiguity gap.\n')
        f.write('* If sequential replay shows lower masked/context/rollout error and higher\n')
        f.write('  contiguity gap, that supports the claim that replay consolidates temporal\n')
        f.write('  structure rather than only itemwise reconstruction.\n')
        f.write(f"\nSequential direct latent-transition MSE: {results['seq_direct_transition_mse']:.6f}\n")

    print('\nResults summary')
    for key in [
        'iid_clean_successor_mse', 'seq_clean_successor_mse',
        'iid_masked_successor_mse', 'seq_masked_successor_mse',
        'iid_masked_context_mse', 'seq_masked_context_mse',
        'iid_context_gain', 'seq_context_gain',
        'iid_rollout_mean_mse', 'seq_rollout_mean_mse',
        'iid_contiguity_gap', 'seq_contiguity_gap',
        'seq_direct_transition_mse'
    ]:
        print(f'  {key}: {results[key]:.6f}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--corruption', type=str, default='mask', choices=['mask', 'noise'])
    parser.add_argument('--run_dir', type=str, default=None)
    args = parser.parse_args()
    main(seed=args.seed, n_samples=args.n_samples, corruption=args.corruption, run_dir=args.run_dir)
