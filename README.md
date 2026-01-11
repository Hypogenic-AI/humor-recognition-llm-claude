# Low-Rank Humor Recognition in LLMs

This research investigates the linear representation of humor in large language models and measures its effective dimensionality.

## Key Findings

- **Humor is linearly separable**: A linear probe achieves **94.1% accuracy** at distinguishing humor vs. non-humor in GPT-2's hidden representations
- **The representation is low-rank**: Only **4 dimensions** achieve 90% of best accuracy; **15 dimensions** achieve 95%
- **A single direction captures most signal**: PCA component 4 alone achieves **78.9% accuracy**
- **Best layer is 7** (middle-to-late), consistent with prior work on sentiment/truth

## Quick Results

| Metric | Value |
|--------|-------|
| Best probe accuracy | 94.1% |
| Best layer | 7 (of 12) |
| Dimensions for 90% accuracy | 4 |
| Dimensions for 95% accuracy | 15 |
| Single-component max accuracy | 78.9% |

## Repository Structure

```
.
├── REPORT.md                  # Full research report
├── README.md                  # This file
├── planning.md                # Experimental design
├── literature_review.md       # Background literature
├── resources.md               # Resource catalog
├── src/
│   ├── experiment.py          # Main experiment code
│   └── rank_analysis_detailed.py  # Detailed rank analysis
├── results/
│   ├── probe_results.csv      # Per-layer probing results
│   ├── rank_analysis.json     # Summary statistics
│   └── detailed_rank_analysis.json  # Comprehensive rank data
├── figures/
│   ├── probe_accuracy_by_layer.png
│   ├── rank_analysis.png
│   ├── detailed_rank_analysis.png
│   ├── pca_2d_layer7.png
│   └── direction_similarity.png
├── datasets/                  # Dataset storage (not in git)
├── papers/                    # Reference papers
└── code/                      # Reference implementations
```

## Reproducing Results

### Environment Setup

```bash
# Create fresh virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add torch transformers transformer-lens datasets scikit-learn numpy pandas matplotlib seaborn tqdm einops jaxtyping
```

### Run Experiments

```bash
# Main experiment (linear probing + basic rank analysis)
python src/experiment.py

# Detailed rank analysis
python src/rank_analysis_detailed.py
```

### Expected Runtime
- Main experiment: ~10 minutes (GPU)
- Detailed rank analysis: ~5 minutes (GPU)

## Methodology

1. **Data**: ColBERT Humor Detection dataset (200k samples, balanced)
2. **Model**: GPT-2-small (124M params, 12 layers, 768 hidden dim)
3. **Approach**:
   - Extract residual stream activations at final token position
   - Train logistic regression probes at each layer
   - Apply PCA and test classification with reduced dimensions
4. **Metrics**: Accuracy, F1, AUC, explained variance, rank analysis

## Citation

If you use this work, please cite:

```
@misc{humor-rank-2026,
  title={How Low-Rank is Humor Recognition in LLMs?},
  year={2026},
  note={Experimental investigation of humor representation dimensionality}
}
```

## Related Work

- Tigges et al. (2023) - Linear Representations of Sentiment
- Marks & Tegmark (2024) - Geometry of Truth
- Hu et al. (2021) - LoRA: Low-Rank Adaptation

See `literature_review.md` for comprehensive background.

## License

Research code for academic use.
