# Literature Review: How Low-Rank is Humor Recognition in LLMs?

## Research Question

**Hypothesis**: There exists a basis in the hidden representations of large language models (LLMs) for humor recognition, and the rank of this basis is low.

This literature review synthesizes research on: (1) humor detection in NLP/LLMs, (2) linear representations in LLMs, and (3) low-rank structures in neural network adaptations.

---

## 1. Research Area Overview

The intersection of humor recognition and LLM interpretability is an emerging area that combines computational humor analysis with mechanistic interpretability. Recent work has shown that LLMs encode many semantic concepts as linear directions in activation space (the "Linear Representation Hypothesis"). This opens the possibility that humor recognition, like sentiment, truth, and other features, may be represented as a low-dimensional subspace within LLM hidden representations.

---

## 2. Key Papers

### 2.1 Linear Representations of Sentiment in Large Language Models
**Authors**: Tigges, Hollinsworth, Geiger, Nanda (2023)
**arXiv**: 2310.15154
**File**: `papers/2310.15154_linear_sentiment_representations.pdf`

**Key Contribution**: Demonstrates that sentiment is represented linearly in LLMs - a single direction in activation space captures positive/negative sentiment across diverse tasks.

**Methodology**:
- Five methods to find sentiment direction: Mean Difference, K-means, Linear Probing, DAS, and PCA
- All methods converge to similar directions (cosine similarity 70-99%)
- Causal interventions via activation patching and ablation

**Key Findings**:
- Sentiment is encoded as a single direction in the residual stream
- This direction is causally relevant - ablating it reduces classification accuracy by 76%
- "Summarization motif" discovered: sentiment aggregates at punctuation tokens
- Direction generalizes across intermediate layers (best at middle layers)

**Relevance to Our Research**:
This is the most directly relevant paper. If sentiment has a linear representation, humor might too. Key questions:
- Is humor similarly linear?
- What is the dimensionality (rank) of humor representation?
- Where in the model does humor representation emerge?

**Datasets Used**: ToyMovieReview (synthetic), Stanford Sentiment Treebank
**Models**: GPT-2, Pythia (70M to 2.8B)
**Code**: Not publicly available but methodology is reproducible

---

### 2.2 The Geometry of Truth: Emergent Linear Structure in LLM Representations
**Authors**: Marks, Tegmark (2024)
**arXiv**: 2310.06824
**File**: `papers/2310.06824_geometry_of_truth.pdf`

**Key Contribution**: Shows that LLMs linearly represent truth/falsehood of factual statements.

**Methodology**:
- Curated high-quality true/false datasets
- PCA visualization showing clear linear structure
- Transfer experiments: probes trained on one dataset generalize
- Causal interventions: surgical modifications flip truth predictions

**Key Findings**:
- Linear representations of truth emerge with scale
- Simple difference-in-mean probes work as well as complex probing techniques
- Truth direction localizes to specific hidden states at specific token positions
- Larger models have more abstract, generalizable truth representations

**Relevance to Our Research**:
- Methodology is directly applicable to humor
- Probing and visualization techniques can be replicated
- Demonstrates that abstract semantic features have linear structure

**Code**: https://github.com/saprmarks/geometry-of-truth

---

### 2.3 Sparse Autoencoders Find Highly Interpretable Features in Language Models
**Authors**: Cunningham, Ewart, Riggs, Huben, Sharkey (2023)
**arXiv**: 2309.08600
**File**: `papers/2309.08600_sparse_autoencoders_interpretability.pdf`

**Key Contribution**: Demonstrates sparse autoencoders (SAEs) can extract interpretable features from LLM activations.

**Methodology**:
- Train sparse autoencoders on LLM internal activations
- Sparsity penalty encourages monosemantic features
- Evaluate interpretability using automated scoring (GPT-4)

**Key Findings**:
- SAE features are more interpretable than neurons, PCA, or ICA
- Features can pinpoint causally-important directions for specific tasks
- Addresses polysemanticity through sparse coding

**Relevance to Our Research**:
- SAEs could identify humor-related features
- Method for quantifying interpretability of discovered features
- Could reveal how many independent directions contribute to humor

**Code**: https://github.com/HoagyC/sparse_coding, SAELens library

---

### 2.4 LoRA: Low-Rank Adaptation of Large Language Models
**Authors**: Hu et al. (2021)
**arXiv**: 2106.09685
**File**: `papers/2106.09685_LoRA_low_rank_adaptation.pdf`

**Key Contribution**: Shows task-specific adaptations have low intrinsic rank.

**Methodology**:
- Inject low-rank decomposition matrices (BA where B and A are low-rank)
- Keep pre-trained weights frozen
- Analyze rank requirements for different tasks

**Key Findings**:
- Pre-trained models have low "intrinsic dimension"
- Updates during adaptation have low "intrinsic rank"
- Even rank r=1 or r=2 suffices for many tasks
- Performance matches full fine-tuning with <0.01% parameters

**Relevance to Our Research**:
- Provides theoretical foundation: if task adaptation is low-rank, task representations might be too
- Methodology for analyzing rank requirements
- Suggests humor classification might require only a few dimensions

**Code**: https://github.com/microsoft/LoRA

---

### 2.5 ColBERT: Using BERT Sentence Embedding for Humor Detection
**Authors**: Annamoradnejad, Zoghi (2020)
**arXiv**: 2004.12765
**File**: `papers/2004.12765_ColBERT_humor_detection.pdf`

**Key Contribution**: Proposes architecture for humor detection based on incongruity theory.

**Methodology**:
- Separate sentence embeddings using BERT
- Parallel neural network paths for each sentence
- Concatenate to detect incongruity/punchline relationships
- 200k dataset (100k humorous, 100k non-humorous)

**Key Findings**:
- F1 score of 98.2% on balanced dataset
- Sentence embeddings + linguistic structure is key
- Humor detection benefits from understanding inter-sentence relationships

**Relevance to Our Research**:
- Provides baseline for humor detection performance
- Dataset is publicly available on HuggingFace
- Suggests incongruity detection is key mechanism

**Code**: https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection

---

### 2.6 A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models
**Authors**: Rai et al. (2024)
**arXiv**: 2407.02646
**File**: `papers/2407.02646_mechanistic_interpretability_review.pdf`

**Key Contribution**: Comprehensive survey of mechanistic interpretability techniques.

**Methodology**: Review of probing, activation patching, SAEs, and circuit analysis.

**Key Findings**:
- Linear representation hypothesis is well-supported across many features
- Sparse autoencoders effectively decompose polysemantic neurons
- Evaluation of interpretability remains challenging

**Relevance to Our Research**:
- Provides methodological toolkit
- Lists evaluation approaches for interpretability claims
- Identifies gaps and best practices

---

### 2.7 Humor Detection: A Transformer Gets the Last Laugh
**Authors**: Weller, Seppi (2019)
**arXiv**: 1909.00252
**File**: `papers/1909.00252_humor_transformer.pdf`

**Key Contribution**: Early application of transformers to humor detection.

**Relevance**: Historical context for humor detection with transformers.

---

### 2.8 Getting Serious about Humor: Crafting Humor Datasets with LLMs
**Authors**: (2024)
**arXiv**: 2403.00794
**File**: `papers/2403.00794_getting_serious_humor_datasets.pdf`

**Key Contribution**: Uses LLMs to generate synthetic humor/non-humor pairs.

**Key Findings**:
- LLMs can effectively "unfun" jokes
- Synthetic data improves humor detection models
- Addresses data scarcity in humor detection

**Relevance to Our Research**:
- Method for generating more training data
- Shows LLMs have internal understanding of humor/non-humor distinction

---

## 3. Common Methodologies

### 3.1 Linear Probing
Used in: Sentiment, Truth, general interpretability research
- Train linear classifier on hidden representations
- If linear probe succeeds, feature is linearly represented
- Different probe types: logistic regression, mean difference, PCA

### 3.2 Activation Patching
Used in: Sentiment, Truth papers
- Swap activations between inputs with different labels
- Measure causal effect on model output
- Localizes important hidden states

### 3.3 Sparse Autoencoders (SAEs)
Used in: Feature discovery, polysemanticity resolution
- Learn overcomplete sparse dictionary of directions
- Each direction ideally corresponds to one interpretable feature
- Can discover unknown features

### 3.4 Dimensionality Analysis
Used in: LoRA, intrinsic dimension literature
- PCA/SVD to analyze rank of representations
- Determine minimum dimensions needed for task

---

## 4. Standard Baselines

For humor detection:
- BERT-based classifiers (F1 ~98% on ColBERT dataset)
- Random baseline: 50% on balanced binary classification
- Bag-of-words/N-gram models: ~75-85%

For linear representation analysis:
- Random direction baseline
- PCA components
- Independent Component Analysis (ICA)
- Individual neurons

---

## 5. Evaluation Metrics

### For Humor Detection:
- Accuracy, F1-score, AUC-ROC
- Human evaluation of generated humor

### For Linear Representations:
- Probe classification accuracy
- Transfer accuracy (train on A, test on B)
- Causal metrics: logit difference, logit flip percentage
- Autointerpretability scores (GPT-4 evaluation)

### For Low-Rank Analysis:
- Explained variance by top-k components
- Performance vs. rank curve
- Intrinsic dimension estimates

---

## 6. Datasets in the Literature

| Dataset | Size | Task | Used In |
|---------|------|------|---------|
| ColBERT | 200k | Binary humor classification | ColBERT paper |
| r/Jokes | 550k+ | Humor with scores | Weller & Seppi |
| Stanford Sentiment Treebank | 10k | Sentiment | Sentiment paper |
| True/False datasets | ~50k | Truth classification | Geometry of Truth |

---

## 7. Gaps and Opportunities

### Gap 1: No Study of Linear Humor Representations
While sentiment and truth have been shown to be linear, no work has systematically investigated whether humor is linearly represented.

### Gap 2: Rank of Humor Representation Unknown
The intrinsic dimensionality of humor recognition in LLMs has not been measured.

### Gap 3: No Causal Analysis of Humor Features
Activation patching and causal interventions have not been applied to humor.

### Gap 4: No SAE Analysis of Humor
Sparse autoencoders haven't been used to discover humor-related features.

---

## 8. Recommendations for Our Experiment

### 8.1 Recommended Datasets

**Primary**: ColBERT Humor Detection (200k samples)
- Balanced binary classification
- Short texts suitable for probing
- Available on HuggingFace

**Secondary**: Short Jokes (231k samples)
- All jokes (positive class)
- Can pair with news headlines as negative class

### 8.2 Recommended Baselines

1. Linear probe (logistic regression)
2. Mean difference direction
3. PCA-based direction
4. Random direction (null baseline)

### 8.3 Recommended Metrics

1. Probe classification accuracy
2. PCA explained variance analysis
3. Rank sweep (performance vs. number of components)
4. Causal metrics via activation patching

### 8.4 Recommended Models

- GPT-2 (small, medium) - well-understood
- Pythia (70M, 160M, 410M, 1.4B) - good scale analysis
- LLaMA-2 (7B, 13B) - for larger models if compute allows

### 8.5 Methodological Considerations

1. **Start simple**: Begin with linear probing on final token position
2. **Layer analysis**: Check which layers contain humor representations
3. **Token position analysis**: Check summarization behavior (period, punchline tokens)
4. **Rank analysis**: Use PCA to measure effective dimensionality
5. **Causal validation**: Use activation patching to verify causality
6. **Cross-dataset transfer**: Train on one dataset, test on another

---

## 9. Key References

1. Tigges et al. (2023) - Linear Representations of Sentiment
2. Marks & Tegmark (2024) - Geometry of Truth
3. Cunningham et al. (2023) - Sparse Autoencoders
4. Hu et al. (2021) - LoRA
5. Annamoradnejad & Zoghi (2020) - ColBERT
6. Rai et al. (2024) - MI Review

---

## 10. Summary

The literature strongly supports the hypothesis that semantic features like sentiment and truth are linearly represented in LLMs. The LoRA paper demonstrates that task-specific adaptations are low-rank. However, no study has investigated whether humor recognition follows this pattern.

Our experiment should:
1. Test whether humor is linearly separable in LLM hidden spaces
2. Measure the rank/dimensionality of humor representations
3. Identify which layers and token positions encode humor
4. Validate findings with causal interventions

The methodological toolkit (linear probing, PCA, activation patching, SAEs) is well-established. The datasets (ColBERT, short jokes) are readily available. The research question is novel and addresses a clear gap in the literature.
