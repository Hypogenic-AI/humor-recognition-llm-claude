#!/usr/bin/env python3
"""
Main experiment script for investigating low-rank humor representations in LLMs.

Research Question: Is there a low-rank linear basis for humor recognition
in LLM hidden representations?

Methodology:
1. Extract activations from GPT-2-small for humor/non-humor texts
2. Train linear probes at each layer
3. Analyze rank via PCA
4. Validate with multiple probing methods
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "gpt2"  # GPT-2 small (124M params)
BATCH_SIZE = 32
MAX_LENGTH = 128
N_SAMPLES = 10000  # Use subset for efficiency (can increase if time permits)
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# Data Loading
# =============================================================================

def load_humor_dataset(n_samples: int = N_SAMPLES) -> Dict[str, Dataset]:
    """
    Load the ColBERT Humor Detection dataset from HuggingFace.

    Returns:
        Dict with train, val, test splits
    """
    print("Loading ColBERT Humor Detection dataset...")
    dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")

    # Get the train split and shuffle
    data = dataset['train'].shuffle(seed=SEED)

    # Take a subset for efficiency
    if n_samples and n_samples < len(data):
        data = data.select(range(n_samples))

    # Create splits: 80% train, 10% val, 10% test
    n = len(data)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    splits = {
        'train': data.select(range(train_size)),
        'val': data.select(range(train_size, train_size + val_size)),
        'test': data.select(range(train_size + val_size, n))
    }

    print(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Check class balance
    for split_name, split_data in splits.items():
        humor_count = sum(1 for ex in split_data if ex['humor'])
        print(f"  {split_name}: {humor_count}/{len(split_data)} humor ({100*humor_count/len(split_data):.1f}%)")

    return splits

# =============================================================================
# Activation Extraction
# =============================================================================

def load_model() -> Tuple[HookedTransformer, torch.device]:
    """Load GPT-2 model with TransformerLens."""
    print(f"\nLoading {MODEL_NAME} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    print(f"Model loaded: {MODEL_NAME}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print(f"  Vocabulary: {model.cfg.d_vocab}")

    return model, device


def extract_activations(
    model: HookedTransformer,
    texts: List[str],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    layer_hook: str = "resid_post"  # residual stream after layer
) -> Dict[int, np.ndarray]:
    """
    Extract hidden state activations from all layers.

    We extract the activation at the final token position, following the
    methodology from the sentiment paper (Tigges et al. 2023).

    Args:
        model: The transformer model
        texts: List of input texts
        device: Computation device
        batch_size: Batch size for inference
        layer_hook: Which activation to extract

    Returns:
        Dict mapping layer index to activations array (n_samples, d_model)
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Initialize storage
    all_activations = {layer: [] for layer in range(n_layers)}

    print(f"\nExtracting activations from {len(texts)} samples...")

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        tokens = model.to_tokens(batch_texts, prepend_bos=True)

        # Get sequence lengths (position of last non-padding token)
        # For now, we'll use the last token position for each sequence
        seq_lengths = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1

        # Run forward pass with hooks
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract activations at final token position for each layer
        for layer in range(n_layers):
            # Get residual stream after this layer
            hook_name = f"blocks.{layer}.hook_{layer_hook}"
            activations = cache[hook_name]  # (batch, seq_len, d_model)

            # Get activation at final token for each sample
            final_acts = []
            for j in range(len(batch_texts)):
                final_pos = min(seq_lengths[j].item(), activations.shape[1] - 1)
                final_acts.append(activations[j, final_pos, :].cpu().numpy())

            all_activations[layer].extend(final_acts)

        # Clear cache to save memory
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to numpy arrays
    for layer in range(n_layers):
        all_activations[layer] = np.array(all_activations[layer])
        print(f"Layer {layer}: {all_activations[layer].shape}")

    return all_activations

# =============================================================================
# Linear Probing
# =============================================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regularization: float = 1.0
) -> Dict:
    """
    Train a logistic regression probe and evaluate.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        regularization: L2 regularization strength (1/C)

    Returns:
        Dict with accuracy, f1, auc, and probe weights
    """
    probe = LogisticRegression(
        C=1.0/regularization,
        max_iter=1000,
        random_state=SEED,
        solver='lbfgs'
    )

    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'weights': probe.coef_[0],
        'bias': probe.intercept_[0]
    }


def compute_mean_difference_direction(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Compute the mean difference direction: μ_humor - μ_non_humor.

    This is the simplest probing method and provides a baseline direction.
    """
    humor_mask = y == 1
    mean_humor = X[humor_mask].mean(axis=0)
    mean_non_humor = X[~humor_mask].mean(axis=0)

    direction = mean_humor - mean_non_humor
    # Normalize to unit vector
    direction = direction / np.linalg.norm(direction)

    return direction


def probe_all_layers(
    activations_train: Dict[int, np.ndarray],
    activations_test: Dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train probes at each layer and collect results.

    Returns:
        DataFrame with probing results per layer
    """
    results = []

    n_layers = len(activations_train)
    print(f"\nTraining probes at {n_layers} layers...")

    for layer in tqdm(range(n_layers), desc="Probing"):
        X_train = activations_train[layer]
        X_test = activations_test[layer]

        # Logistic regression probe
        probe_result = train_linear_probe(X_train, y_train, X_test, y_test)

        # Mean difference direction
        mean_dir = compute_mean_difference_direction(X_train, y_train)

        # Classify using mean difference direction
        # Project test data onto direction and threshold at 0
        projections = X_test @ mean_dir
        mean_proj = projections.mean()
        y_pred_mean = (projections > mean_proj).astype(int)
        mean_acc = accuracy_score(y_test, y_pred_mean)

        # Compute cosine similarity between probe weights and mean difference
        probe_dir = probe_result['weights'] / np.linalg.norm(probe_result['weights'])
        cosine_sim = np.abs(np.dot(probe_dir, mean_dir))

        # Random baseline
        random_dir = np.random.randn(X_train.shape[1])
        random_dir = random_dir / np.linalg.norm(random_dir)
        random_proj = X_test @ random_dir
        y_pred_random = (random_proj > random_proj.mean()).astype(int)
        random_acc = accuracy_score(y_test, y_pred_random)

        results.append({
            'layer': layer,
            'probe_accuracy': probe_result['accuracy'],
            'probe_f1': probe_result['f1'],
            'probe_auc': probe_result['auc'],
            'mean_diff_accuracy': mean_acc,
            'random_accuracy': random_acc,
            'direction_cosine_sim': cosine_sim
        })

    return pd.DataFrame(results)

# =============================================================================
# Rank Analysis (PCA)
# =============================================================================

def analyze_rank(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 100
) -> Dict:
    """
    Analyze the effective rank of humor representation using PCA.

    We analyze:
    1. Explained variance by top-k components
    2. Classification accuracy vs. number of components
    3. Effective dimensionality estimates

    Args:
        X: Activation matrix (n_samples, d_model)
        y: Labels
        max_components: Maximum components to analyze

    Returns:
        Dict with rank analysis results
    """
    n_components = min(max_components, X.shape[1], X.shape[0])

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X)

    # Explained variance analysis
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Find effective rank at different thresholds
    rank_90 = np.searchsorted(cumulative_var, 0.90) + 1
    rank_95 = np.searchsorted(cumulative_var, 0.95) + 1
    rank_99 = np.searchsorted(cumulative_var, 0.99) + 1

    # Classification accuracy vs. number of components
    accs_by_rank = []
    ranks_to_test = [1, 2, 3, 5, 10, 20, 50, 100]
    ranks_to_test = [r for r in ranks_to_test if r <= n_components]

    # Cross-validation for more robust estimates
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for r in ranks_to_test:
        X_reduced = X_pca[:, :r]
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=SEED),
            X_reduced, y,
            cv=cv,
            scoring='accuracy'
        )
        accs_by_rank.append({
            'rank': r,
            'accuracy': scores.mean(),
            'std': scores.std()
        })

    return {
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'rank_90': int(rank_90),
        'rank_95': int(rank_95),
        'rank_99': int(rank_99),
        'accuracy_by_rank': pd.DataFrame(accs_by_rank),
        'pca_components': pca.components_
    }


def analyze_class_separation(
    X: np.ndarray,
    y: np.ndarray
) -> Dict:
    """
    Analyze how humor and non-humor classes separate in activation space.

    Computes:
    - Mean difference norm
    - Class variance
    - Fisher discriminant ratio
    """
    humor_mask = y == 1
    X_humor = X[humor_mask]
    X_non_humor = X[~humor_mask]

    mean_humor = X_humor.mean(axis=0)
    mean_non_humor = X_non_humor.mean(axis=0)

    # Between-class variance (mean difference)
    between_class = mean_humor - mean_non_humor
    between_norm = np.linalg.norm(between_class)

    # Within-class variance
    var_humor = np.var(X_humor, axis=0).mean()
    var_non_humor = np.var(X_non_humor, axis=0).mean()
    within_var = (var_humor + var_non_humor) / 2

    # Fisher discriminant ratio
    fisher_ratio = between_norm**2 / within_var if within_var > 0 else 0

    return {
        'between_class_norm': between_norm,
        'within_class_var': within_var,
        'fisher_ratio': fisher_ratio,
        'mean_humor': mean_humor,
        'mean_non_humor': mean_non_humor
    }

# =============================================================================
# Visualization
# =============================================================================

def plot_probe_accuracy_by_layer(results_df: pd.DataFrame, save_path: Path):
    """Plot probe accuracy across layers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['layer'], results_df['probe_accuracy'],
            'b-o', label='Logistic Probe', linewidth=2, markersize=8)
    ax.plot(results_df['layer'], results_df['mean_diff_accuracy'],
            'g-s', label='Mean Difference', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Humor Classification Accuracy by Layer (GPT-2 Small)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rank_analysis(rank_results: Dict, save_path: Path):
    """Plot rank analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Explained variance
    ax = axes[0]
    n_comp = len(rank_results['cumulative_variance'])
    ax.bar(range(1, min(21, n_comp+1)),
           rank_results['explained_variance'][:20],
           alpha=0.7, label='Individual')
    ax.plot(range(1, min(21, n_comp+1)),
            rank_results['cumulative_variance'][:20],
            'r-o', label='Cumulative', linewidth=2)
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7,
               label=f'90% (rank={rank_results["rank_90"]})')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7,
               label=f'95% (rank={rank_results["rank_95"]})')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('PCA Explained Variance', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Accuracy vs. rank
    ax = axes[1]
    acc_df = rank_results['accuracy_by_rank']
    ax.errorbar(acc_df['rank'], acc_df['accuracy'],
                yerr=acc_df['std'], fmt='b-o', capsize=4,
                linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)
    ax.set_xlabel('Number of PCA Components (Rank)', fontsize=12)
    ax.set_ylabel('5-Fold CV Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy vs. Representation Rank', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_direction_similarity(results_df: pd.DataFrame, save_path: Path):
    """Plot cosine similarity between probing methods across layers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['layer'], results_df['direction_cosine_sim'],
            'purple', marker='o', linewidth=2, markersize=8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Similarity: Logistic Probe vs Mean Difference Direction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pca_2d(X: np.ndarray, y: np.ndarray, save_path: Path, title: str = ""):
    """Plot 2D PCA projection of activations."""
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    humor_mask = y == 1
    ax.scatter(X_2d[~humor_mask, 0], X_2d[~humor_mask, 1],
               alpha=0.5, c='blue', label='Non-Humor', s=20)
    ax.scatter(X_2d[humor_mask, 0], X_2d[humor_mask, 1],
               alpha=0.5, c='red', label='Humor', s=20)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'2D PCA of Activations {title}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the full experiment pipeline."""
    set_seed(SEED)

    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Experiment configuration
    config = {
        'seed': SEED,
        'model': MODEL_NAME,
        'n_samples': N_SAMPLES,
        'batch_size': BATCH_SIZE,
        'max_length': MAX_LENGTH,
        'timestamp': datetime.now().isoformat()
    }

    print("="*60)
    print("Experiment: Low-Rank Humor Recognition in LLMs")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print("="*60)

    splits = load_humor_dataset(n_samples=N_SAMPLES)

    # Extract texts and labels
    train_texts = [ex['text'] for ex in splits['train']]
    train_labels = np.array([1 if ex['humor'] else 0 for ex in splits['train']])

    test_texts = [ex['text'] for ex in splits['test']]
    test_labels = np.array([1 if ex['humor'] else 0 for ex in splits['test']])

    # Show sample data
    humor_indices = np.where(train_labels == 1)[0]
    non_humor_indices = np.where(train_labels == 0)[0]
    print("\nSample humor text:", train_texts[humor_indices[0]])
    print("Sample non-humor text:", train_texts[non_humor_indices[0]])

    # =========================================================================
    # Step 2: Load Model and Extract Activations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 2: Extracting Activations")
    print("="*60)

    model, device = load_model()

    # Extract activations
    train_activations = extract_activations(model, train_texts, device)
    test_activations = extract_activations(model, test_texts, device)

    # =========================================================================
    # Step 3: Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("Step 3: Linear Probing")
    print("="*60)

    probe_results = probe_all_layers(
        train_activations, test_activations,
        train_labels, test_labels
    )

    print("\nProbe Results Summary:")
    print(probe_results.to_string(index=False))

    # Find best layer
    best_layer = probe_results.loc[probe_results['probe_accuracy'].idxmax()]
    print(f"\nBest layer: {int(best_layer['layer'])}")
    print(f"  Accuracy: {best_layer['probe_accuracy']:.4f}")
    print(f"  F1: {best_layer['probe_f1']:.4f}")
    print(f"  AUC: {best_layer['probe_auc']:.4f}")

    # =========================================================================
    # Step 4: Rank Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("Step 4: Rank Analysis")
    print("="*60)

    best_layer_idx = int(best_layer['layer'])

    # Combine train and test for PCA analysis
    X_all = np.vstack([train_activations[best_layer_idx],
                       test_activations[best_layer_idx]])
    y_all = np.concatenate([train_labels, test_labels])

    rank_results = analyze_rank(X_all, y_all)

    print(f"\nRank Analysis at Layer {best_layer_idx}:")
    print(f"  Rank for 90% variance: {rank_results['rank_90']}")
    print(f"  Rank for 95% variance: {rank_results['rank_95']}")
    print(f"  Rank for 99% variance: {rank_results['rank_99']}")

    print("\nAccuracy vs. Rank:")
    print(rank_results['accuracy_by_rank'].to_string(index=False))

    # Class separation analysis
    separation = analyze_class_separation(X_all, y_all)
    print(f"\nClass Separation:")
    print(f"  Between-class norm: {separation['between_class_norm']:.4f}")
    print(f"  Within-class variance: {separation['within_class_var']:.4f}")
    print(f"  Fisher ratio: {separation['fisher_ratio']:.4f}")

    # =========================================================================
    # Step 5: Generate Visualizations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 5: Generating Visualizations")
    print("="*60)

    plot_probe_accuracy_by_layer(probe_results, FIGURES_DIR / "probe_accuracy_by_layer.png")
    plot_rank_analysis(rank_results, FIGURES_DIR / "rank_analysis.png")
    plot_direction_similarity(probe_results, FIGURES_DIR / "direction_similarity.png")
    plot_pca_2d(X_all, y_all, FIGURES_DIR / f"pca_2d_layer{best_layer_idx}.png",
                title=f"(Layer {best_layer_idx})")

    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print("\n" + "="*60)
    print("Step 6: Saving Results")
    print("="*60)

    # Save config
    with open(RESULTS_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save probe results
    probe_results.to_csv(RESULTS_DIR / "probe_results.csv", index=False)

    # Save rank analysis
    rank_summary = {
        'best_layer': best_layer_idx,
        'best_probe_accuracy': float(best_layer['probe_accuracy']),
        'best_probe_f1': float(best_layer['probe_f1']),
        'best_probe_auc': float(best_layer['probe_auc']),
        'rank_90': rank_results['rank_90'],
        'rank_95': rank_results['rank_95'],
        'rank_99': rank_results['rank_99'],
        'between_class_norm': float(separation['between_class_norm']),
        'within_class_var': float(separation['within_class_var']),
        'fisher_ratio': float(separation['fisher_ratio']),
        'accuracy_by_rank': rank_results['accuracy_by_rank'].to_dict('records')
    }

    with open(RESULTS_DIR / "rank_analysis.json", 'w') as f:
        json.dump(rank_summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)

    print("\nKey Findings:")
    print(f"1. Best linear probe accuracy: {best_layer['probe_accuracy']:.1%} at layer {best_layer_idx}")
    print(f"   (Random baseline: 50%)")
    print(f"2. Effective rank for 90% variance: {rank_results['rank_90']} dimensions")
    print(f"3. Effective rank for 95% variance: {rank_results['rank_95']} dimensions")
    print(f"4. Humor/non-humor are linearly separable: {'Yes' if best_layer['probe_accuracy'] > 0.6 else 'Partially'}")

    return {
        'config': config,
        'probe_results': probe_results,
        'rank_results': rank_summary,
        'best_layer': best_layer_idx
    }


if __name__ == "__main__":
    results = run_experiment()
