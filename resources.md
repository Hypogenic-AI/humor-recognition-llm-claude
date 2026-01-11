# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project investigating low-rank humor recognition in LLMs.

**Research Hypothesis**: There exists a basis in the hidden representations of large language models (LLMs) for humor recognition, and the rank of this basis is low.

---

## Papers

**Total papers downloaded: 8**

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Linear Representations of Sentiment in LLMs | Tigges et al. | 2023 | papers/2310.15154_linear_sentiment_representations.pdf | Core methodology for linear probing |
| The Geometry of Truth | Marks, Tegmark | 2024 | papers/2310.06824_geometry_of_truth.pdf | Linear truth representations, code available |
| Sparse Autoencoders Find Interpretable Features | Cunningham et al. | 2023 | papers/2309.08600_sparse_autoencoders_interpretability.pdf | SAE methodology |
| LoRA: Low-Rank Adaptation | Hu et al. | 2021 | papers/2106.09685_LoRA_low_rank_adaptation.pdf | Low-rank adaptation theory |
| ColBERT: BERT Sentence Embedding for Humor | Annamoradnejad, Zoghi | 2020 | papers/2004.12765_ColBERT_humor_detection.pdf | Humor detection baseline |
| Mechanistic Interpretability Review | Rai et al. | 2024 | papers/2407.02646_mechanistic_interpretability_review.pdf | Comprehensive MI survey |
| Humor Detection: A Transformer Gets the Last Laugh | Weller, Seppi | 2019 | papers/1909.00252_humor_transformer.pdf | Early transformer humor work |
| Getting Serious about Humor | Various | 2024 | papers/2403.00794_getting_serious_humor_datasets.pdf | LLM-generated humor data |

See papers/README.md for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 4**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| ColBERT Humor | HuggingFace | 200k | Binary classification | datasets/colbert_humor/ | Primary dataset |
| Short Jokes | HuggingFace | 231k | Joke generation/detection | datasets/short_jokes/ | All positive class |
| Dad Jokes | HuggingFace | 53k | Question-response jokes | datasets/dadjokes/ | Q&A format |
| Reddit Jokes | HuggingFace | 100k subset | Jokes with scores | datasets/reddit_jokes_100k/ | Has humor ratings |

See datasets/README.md for detailed descriptions.

---

## Code Repositories

**Total repositories cloned: 5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ColBERT Humor | github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection | Humor baseline | code/colbert_humor/ | Dataset + model |
| SAELens | github.com/jbloomAus/SAELens | Sparse autoencoders | code/saelens/ | Feature extraction |
| Geometry of Truth | github.com/saprmarks/geometry-of-truth | Linear probing | code/geometry_of_truth/ | Probing methodology |
| LoRA | github.com/microsoft/LoRA | Low-rank adaptation | code/lora/ | LoRA implementation |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Interpretability | code/transformer_lens/ | Activation extraction |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**:
   - Searched arXiv, Semantic Scholar, Papers with Code
   - Keywords: "humor detection LLM", "linear representations", "low-rank neural networks", "mechanistic interpretability", "probing classifiers"
   - Focused on papers from 2019-2025

2. **Dataset Search**:
   - Primary source: HuggingFace Datasets
   - Secondary: Papers with Code, Kaggle
   - Prioritized balanced binary classification datasets

3. **Code Search**:
   - GitHub repositories linked from papers
   - Popular interpretability libraries (TransformerLens, SAELens)
   - Official implementations

### Selection Criteria

- **Papers**: Methodological relevance, recency, reproducibility
- **Datasets**: Size (>50k preferred), task alignment (binary classification), availability
- **Code**: Documentation quality, active maintenance, compatibility with modern libraries

### Challenges Encountered

1. No existing papers directly studying linear humor representations
2. Most humor datasets are joke-only (no negative class)
3. Some older datasets have licensing restrictions

### Gaps and Workarounds

- **Gap**: No labeled humor/non-humor dataset with LLM hidden states
  - **Workaround**: Use ColBERT dataset and extract activations ourselves

- **Gap**: No existing humor probing code
  - **Workaround**: Adapt geometry-of-truth methodology

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset(s)

**ColBERT Humor Detection** (datasets/colbert_humor/)
- 200k samples (100k humor, 100k non-humor)
- Balanced binary classification
- Short texts suitable for single-pass inference
- Well-documented and validated

**Rationale**: Best balance of size, task alignment, and quality. The balanced classes make it ideal for probing experiments.

### 2. Baseline Methods

From the literature, implement these probing approaches:

1. **Mean Difference Direction** (simplest)
   - μ_humor - μ_non_humor

2. **Logistic Regression Probe** (most common)
   - Train linear classifier on activations

3. **PCA Analysis** (for rank estimation)
   - Fit PCA, analyze explained variance

4. **Random Direction** (null baseline)
   - Random unit vector for comparison

### 3. Evaluation Metrics

- **Classification**: Accuracy, F1, AUC-ROC
- **Rank Analysis**:
  - Explained variance by top-k components
  - Performance vs. rank curve
  - Effective dimensionality estimates
- **Causal**: Logit difference after patching

### 4. Code to Adapt/Reuse

| Task | Repository | Key Files |
|------|-----------|-----------|
| Activation extraction | code/transformer_lens/ | HookedTransformer |
| Linear probing | code/geometry_of_truth/ | probing.py |
| Sparse autoencoders | code/saelens/ | sae.py |
| LoRA analysis | code/lora/ | loralib/ |

### 5. Suggested Experiment Pipeline

```
1. Data Preparation
   - Load ColBERT dataset
   - Split into train/val/test
   - Format prompts for LLM

2. Activation Extraction
   - Use TransformerLens to extract hidden states
   - Store activations for all layers
   - Focus on final token position (and punctuation)

3. Linear Probing
   - Train probes at each layer
   - Compare probe methods
   - Identify best layer for humor

4. Rank Analysis
   - PCA on humor/non-humor activations
   - Measure explained variance
   - Determine effective rank

5. Causal Validation
   - Activation patching experiments
   - Verify probed direction is causal

6. SAE Analysis (optional)
   - Train SAE on identified layer
   - Identify humor-related features
```

---

## File Structure

```
humor-recognition-llm-claude/
├── papers/                          # Downloaded PDFs
│   ├── 2310.15154_linear_sentiment_representations.pdf
│   ├── 2310.06824_geometry_of_truth.pdf
│   ├── 2309.08600_sparse_autoencoders_interpretability.pdf
│   ├── 2106.09685_LoRA_low_rank_adaptation.pdf
│   ├── 2004.12765_ColBERT_humor_detection.pdf
│   ├── 2407.02646_mechanistic_interpretability_review.pdf
│   ├── 1909.00252_humor_transformer.pdf
│   └── 2403.00794_getting_serious_humor_datasets.pdf
├── datasets/                        # Downloaded datasets
│   ├── colbert_humor/              # Primary dataset (200k)
│   ├── short_jokes/                # 231k jokes
│   ├── dadjokes/                   # 53k dad jokes
│   ├── reddit_jokes_100k/          # 100k reddit jokes
│   ├── samples.json                # Sample data for reference
│   └── .gitignore                  # Excludes large data files
├── code/                            # Cloned repositories
│   ├── colbert_humor/              # Humor baseline
│   ├── saelens/                    # SAE library
│   ├── geometry_of_truth/          # Probing methodology
│   ├── lora/                       # LoRA implementation
│   └── transformer_lens/           # Activation extraction
├── literature_review.md             # Comprehensive lit review
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker
```

---

## Dependencies to Install

For experiment execution, the following Python packages are needed:

```bash
# Core ML
pip install torch transformers datasets

# Interpretability
pip install transformer-lens sae-lens

# Analysis
pip install scikit-learn numpy pandas matplotlib seaborn

# Utilities
pip install tqdm einops jaxtyping
```

---

## Quick Start for Experiment Runner

1. **Verify datasets are loaded**:
   ```python
   from datasets import load_from_disk
   ds = load_from_disk("datasets/colbert_humor")
   print(f"Dataset size: {len(ds['train'])}")
   ```

2. **Extract activations** (example):
   ```python
   from transformer_lens import HookedTransformer
   model = HookedTransformer.from_pretrained("gpt2")
   # See code/transformer_lens/ for full examples
   ```

3. **Run linear probe** (example):
   ```python
   from sklearn.linear_model import LogisticRegression
   probe = LogisticRegression()
   probe.fit(activations, labels)
   # See code/geometry_of_truth/ for full methodology
   ```

---

## References

All papers are available in the `papers/` directory. Key citations:

- Tigges et al. (2023). Linear Representations of Sentiment. arXiv:2310.15154
- Marks & Tegmark (2024). Geometry of Truth. arXiv:2310.06824
- Cunningham et al. (2023). Sparse Autoencoders. arXiv:2309.08600
- Hu et al. (2021). LoRA. arXiv:2106.09685
- Annamoradnejad & Zoghi (2020). ColBERT. arXiv:2004.12765
