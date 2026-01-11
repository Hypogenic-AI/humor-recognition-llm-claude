# Cloned Repositories

This directory contains code repositories relevant to the humor recognition research project.

## Repository Overview

| Repository | Purpose | Key Files |
|------------|---------|-----------|
| colbert_humor | Humor detection baseline | model architecture, dataset |
| saelens | Sparse autoencoders | SAE training, feature extraction |
| geometry_of_truth | Linear probing methodology | probing, patching experiments |
| lora | Low-rank adaptation | loralib implementation |
| transformer_lens | Activation extraction | HookedTransformer |

---

## Repository 1: ColBERT Humor Detection

- **Location**: `code/colbert_humor/`
- **URL**: https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection
- **Purpose**: Baseline humor detection model and dataset

### Key Files
- `ColBERT.py` - Model architecture
- `Dataprep.py` - Data preparation scripts
- Dataset download links

### Usage
```python
# See repository README for full instructions
# Model uses BERT sentence embeddings with parallel processing
```

### Notes
- Original ColBERT paper implementation
- BERT-based, not suitable for LLM probing
- Useful for baseline comparison

---

## Repository 2: SAELens

- **Location**: `code/saelens/`
- **URL**: https://github.com/jbloomAus/SAELens
- **Purpose**: Train sparse autoencoders on LLM activations

### Key Files
- `sae_lens/training/` - SAE training code
- `sae_lens/sae.py` - SAE architecture
- `sae_lens/toolkit/` - Analysis tools

### Usage
```python
from sae_lens import SAE, SAETrainingRunner

# Load pre-trained SAE
sae = SAE.load_from_pretrained("gpt2-small-res-jb")

# Or train your own
runner = SAETrainingRunner(cfg)
sae = runner.run()
```

### Notes
- Well-maintained library
- Pre-trained SAEs available for common models
- Good documentation

---

## Repository 3: Geometry of Truth

- **Location**: `code/geometry_of_truth/`
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Purpose**: Linear probing and patching methodology

### Key Files
- `probing.py` - Linear probe training
- `patching.py` - Activation patching experiments
- `datasets/` - True/false dataset construction
- `visualization.py` - PCA visualizations

### Usage
```python
# Example linear probing (adapted from repository)
from sklearn.linear_model import LogisticRegression

# Get activations for true/false statements
true_acts = get_activations(model, true_statements)
false_acts = get_activations(model, false_statements)

# Train probe
probe = LogisticRegression()
probe.fit(
    np.vstack([true_acts, false_acts]),
    [1]*len(true_acts) + [0]*len(false_acts)
)

# The probe weights define the "truth direction"
truth_direction = probe.coef_[0]
```

### Notes
- Most relevant for our methodology
- Includes causal intervention code
- Well-documented experiments

---

## Repository 4: LoRA

- **Location**: `code/lora/`
- **URL**: https://github.com/microsoft/LoRA
- **Purpose**: Low-rank adaptation implementation

### Key Files
- `loralib/` - LoRA layer implementations
- `examples/` - Usage examples
- Rank analysis code

### Usage
```python
import loralib as lora

# Replace linear layer with LoRA
lora_layer = lora.Linear(in_features, out_features, r=4)

# Only LoRA parameters are trainable
lora.mark_only_lora_as_trainable(model)
```

### Notes
- Official Microsoft implementation
- Used for understanding rank requirements
- May inform rank analysis methodology

---

## Repository 5: TransformerLens

- **Location**: `code/transformer_lens/`
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Extract and analyze LLM activations

### Key Files
- `transformer_lens/HookedTransformer.py` - Main model class
- `transformer_lens/hook_points.py` - Activation hooks
- `demos/` - Usage examples

### Usage
```python
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2")

# Run with cache to get activations
logits, cache = model.run_with_cache("Hello world")

# Access specific activations
residual_stream = cache["resid_post", 5]  # Layer 5 residual stream
attn_output = cache["attn_out", 3]  # Layer 3 attention output
```

### Notes
- Essential for activation extraction
- Supports many model architectures
- Well-documented API

---

## Recommended Workflow

### 1. Extract Activations (TransformerLens)

```python
from transformer_lens import HookedTransformer
from datasets import load_from_disk

# Load model and data
model = HookedTransformer.from_pretrained("gpt2")
dataset = load_from_disk("../datasets/colbert_humor")

# Extract activations
activations = []
labels = []

for sample in dataset['train'][:1000]:  # Start with subset
    _, cache = model.run_with_cache(sample['text'])
    # Get residual stream at final token, middle layer
    act = cache["resid_post", model.cfg.n_layers // 2][:, -1, :]
    activations.append(act.cpu().numpy())
    labels.append(int(sample['humor']))
```

### 2. Train Linear Probe (geometry_of_truth style)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = np.vstack(activations)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)

print(f"Probe accuracy: {probe.score(X_test, y_test):.3f}")
```

### 3. Analyze Rank (PCA)

```python
from sklearn.decomposition import PCA

# Separate by class
humor_acts = X[y == 1]
nonhumor_acts = X[y == 0]

# Fit PCA
pca = PCA()
pca.fit(X)

# Analyze variance explained
cumvar = np.cumsum(pca.explained_variance_ratio_)
effective_rank = np.argmax(cumvar > 0.95) + 1
print(f"Effective rank (95% variance): {effective_rank}")
```

### 4. (Optional) Train SAE (SAELens)

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    model_name="gpt2",
    hook_point="blocks.6.hook_resid_post",
    d_in=768,
    expansion_factor=8,
    # ... other config
)

runner = SAETrainingRunner(cfg)
sae = runner.run()
```

---

## Dependencies

```bash
# Install all needed packages
pip install torch transformers
pip install transformer-lens
pip install sae-lens
pip install scikit-learn numpy pandas
pip install matplotlib seaborn
pip install einops jaxtyping
```

---

## Notes

- All repositories cloned with `--depth 1` to save space
- Check individual repository READMEs for detailed documentation
- Some repositories may have additional dependencies
