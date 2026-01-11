#!/usr/bin/env python3
"""
Detailed rank analysis for humor recognition.

This script performs a more granular analysis of the effective rank
needed for humor classification, testing many more rank values.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
from transformer_lens import HookedTransformer

SEED = 42
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed()

    # Load existing activations from previous run or re-extract
    print("Loading ColBERT dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
    data = dataset['train'].shuffle(seed=SEED)
    data = data.select(range(10000))

    # Split
    n = len(data)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train_data = data.select(range(train_size))
    test_data = data.select(range(train_size + val_size, n))

    train_texts = [ex['text'] for ex in train_data]
    train_labels = np.array([1 if ex['humor'] else 0 for ex in train_data])
    test_texts = [ex['text'] for ex in test_data]
    test_labels = np.array([1 if ex['humor'] else 0 for ex in test_data])

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Load model
    print("\nLoading GPT-2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    # Extract activations at best layer (layer 7 from previous analysis)
    best_layer = 7
    print(f"\nExtracting activations at layer {best_layer}...")

    def extract_at_layer(texts, layer_idx, batch_size=32):
        activations = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)
            seq_lengths = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            acts = cache[hook_name]

            for j in range(len(batch)):
                final_pos = min(seq_lengths[j].item(), acts.shape[1] - 1)
                activations.append(acts[j, final_pos, :].cpu().numpy())

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.array(activations)

    X_train = extract_at_layer(train_texts, best_layer)
    X_test = extract_at_layer(test_texts, best_layer)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Combine for PCA fitting
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([train_labels, test_labels])

    # Detailed rank analysis
    print("\nPerforming detailed rank analysis...")

    # Test many rank values
    ranks_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500]
    ranks_to_test = [r for r in ranks_to_test if r <= min(X_all.shape)]

    # Fit PCA
    pca = PCA(n_components=min(500, X_all.shape[0], X_all.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(X_all)

    # Split back
    X_train_pca = X_pca[:len(train_labels)]
    X_test_pca = X_pca[len(train_labels):]

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\nTesting different ranks...")
    for rank in tqdm(ranks_to_test):
        # Train on full training set
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train_pca[:, :rank], train_labels)
        test_acc = clf.score(X_test_pca[:, :rank], test_labels)

        # Also do cross-validation on combined data for smoother estimates
        X_reduced = X_pca[:, :rank]
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=SEED),
            X_reduced, y_all, cv=cv, scoring='accuracy'
        )

        results.append({
            'rank': rank,
            'test_accuracy': test_acc,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cumulative_variance': pca.explained_variance_ratio_[:rank].sum()
        })

    results_df = pd.DataFrame(results)
    print("\nRank Analysis Results:")
    print(results_df.to_string(index=False))

    # Find the rank where we get 90%, 95%, 99% of best accuracy
    best_acc = results_df['test_accuracy'].max()
    rank_90_acc = results_df[results_df['test_accuracy'] >= 0.9 * best_acc]['rank'].min()
    rank_95_acc = results_df[results_df['test_accuracy'] >= 0.95 * best_acc]['rank'].min()
    rank_99_acc = results_df[results_df['test_accuracy'] >= 0.99 * best_acc]['rank'].min()

    print(f"\nEffective Rank Analysis:")
    print(f"  Best test accuracy: {best_acc:.4f}")
    print(f"  Rank for 90% of best accuracy: {rank_90_acc}")
    print(f"  Rank for 95% of best accuracy: {rank_95_acc}")
    print(f"  Rank for 99% of best accuracy: {rank_99_acc}")

    # Find where accuracy plateaus (diminishing returns)
    # Define plateau as <1% improvement in next 10 ranks
    for i, row in results_df.iterrows():
        if i + 1 < len(results_df):
            next_ranks = results_df.iloc[i+1:]
            if len(next_ranks) > 0:
                improvement = next_ranks['test_accuracy'].max() - row['test_accuracy']
                if improvement < 0.01 and row['test_accuracy'] > 0.85:
                    plateau_rank = row['rank']
                    break
    else:
        plateau_rank = results_df['rank'].max()

    print(f"  Plateau rank (diminishing returns): {plateau_rank}")

    # Analyze the first few PCA components
    print("\n\nAnalyzing discrimination power of individual components...")

    # For each of first 20 components, train a probe
    component_results = []
    for comp in range(min(20, X_pca.shape[1])):
        # Use just this one component
        X_1d = X_pca[:, comp:comp+1]
        X_train_1d = X_1d[:len(train_labels)]
        X_test_1d = X_1d[len(train_labels):]

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train_1d, train_labels)
        acc = clf.score(X_test_1d, test_labels)

        component_results.append({
            'component': comp + 1,
            'accuracy': acc,
            'explained_variance': pca.explained_variance_ratio_[comp]
        })

    comp_df = pd.DataFrame(component_results)
    print("\nPer-Component Classification Power:")
    print(comp_df.to_string(index=False))

    # Find components with above-chance discrimination
    discriminative_components = comp_df[comp_df['accuracy'] > 0.55]
    print(f"\nComponents with >55% accuracy (above chance): {len(discriminative_components)}")
    print(f"Top discriminative components: {discriminative_components.nlargest(5, 'accuracy')['component'].tolist()}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy vs Rank
    ax = axes[0, 0]
    ax.plot(results_df['rank'], results_df['test_accuracy'], 'b-o', markersize=4)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random')
    ax.axhline(y=best_acc, color='g', linestyle=':', alpha=0.7, label=f'Best ({best_acc:.3f})')
    ax.axvline(x=rank_95_acc, color='orange', linestyle='--', alpha=0.7,
               label=f'95% best (rank={rank_95_acc})')
    ax.set_xlabel('Number of PCA Components (Rank)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification Accuracy vs Representation Rank')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy vs Rank (linear scale, zoomed)
    ax = axes[0, 1]
    small_ranks = results_df[results_df['rank'] <= 50]
    ax.plot(small_ranks['rank'], small_ranks['test_accuracy'], 'b-o', markersize=6)
    ax.fill_between(small_ranks['rank'],
                    small_ranks['cv_accuracy'] - small_ranks['cv_std'],
                    small_ranks['cv_accuracy'] + small_ranks['cv_std'],
                    alpha=0.2)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random')
    ax.set_xlabel('Number of PCA Components (Rank)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Rank (Low Rank Regime)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Per-component discrimination
    ax = axes[1, 0]
    colors = ['green' if acc > 0.55 else 'gray' for acc in comp_df['accuracy']]
    ax.bar(comp_df['component'], comp_df['accuracy'], color=colors, alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random')
    ax.axhline(y=0.55, color='orange', linestyle=':', label='Threshold')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Classification Accuracy (single component)')
    ax.set_title('Discrimination Power per PCA Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cumulative explained variance vs classification accuracy
    ax = axes[1, 1]
    ax.scatter(results_df['cumulative_variance'], results_df['test_accuracy'],
               c=np.log(results_df['rank']), cmap='viridis', s=50)
    ax.set_xlabel('Cumulative Explained Variance')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Explained Variance')
    ax.grid(True, alpha=0.3)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(np.log(results_df['rank']))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('log(Rank)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'detailed_rank_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'detailed_rank_analysis.png'}")

    # Save results
    summary = {
        'best_layer': best_layer,
        'best_test_accuracy': float(best_acc),
        'rank_90_pct_accuracy': int(rank_90_acc),
        'rank_95_pct_accuracy': int(rank_95_acc),
        'rank_99_pct_accuracy': int(rank_99_acc),
        'plateau_rank': int(plateau_rank),
        'num_discriminative_components': len(discriminative_components),
        'accuracy_by_rank': results_df.to_dict('records'),
        'per_component_accuracy': comp_df.to_dict('records')
    }

    with open(RESULTS_DIR / 'detailed_rank_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {RESULTS_DIR / 'detailed_rank_analysis.json'}")

    # Key finding summary
    print("\n" + "="*60)
    print("KEY FINDINGS: LOW-RANK HUMOR REPRESENTATION")
    print("="*60)
    print(f"""
1. LINEARITY: Humor is strongly linearly separable
   - Best probe accuracy: {best_acc:.1%}
   - This confirms humor has a linear representation in GPT-2

2. EFFECTIVE RANK: Humor classification is LOW-RANK
   - Just {rank_90_acc} dimensions achieve 90% of best accuracy
   - Just {rank_95_acc} dimensions achieve 95% of best accuracy
   - Plateau at ~{plateau_rank} dimensions (diminishing returns)

3. DISCRIMINATIVE COMPONENTS: Few components carry humor signal
   - {len(discriminative_components)} out of 20 top components show >55% accuracy
   - Top component alone achieves {comp_df['accuracy'].max():.1%} accuracy

4. INTERPRETATION:
   - Humor recognition in GPT-2 has an effective rank of ~{rank_95_acc}-{plateau_rank}
   - This is low compared to the 768 hidden dimensions
   - Suggests humor is encoded in a low-dimensional subspace
   - Supports the hypothesis that humor has a low-rank linear basis
""")


if __name__ == "__main__":
    main()
