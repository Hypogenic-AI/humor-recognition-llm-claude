# Research Plan: How Low-Rank is Humor Recognition in LLMs?

## Research Question

**Main Hypothesis**: There exists a basis in the hidden representations of large language models (LLMs) for humor recognition, and the rank of this basis is low.

**Specific Questions**:
1. Can humor be linearly separated in LLM hidden representations?
2. What is the effective dimensionality (rank) of humor representations?
3. Which layers of the model encode humor most effectively?
4. Does the humor direction generalize across different probing methods?

## Background and Motivation

Recent work on mechanistic interpretability has shown that LLMs encode semantic concepts (sentiment, truth) as linear directions in activation space. The LoRA paper demonstrates that task adaptations have low intrinsic rank. This raises a natural question: does humor recognition follow the same pattern?

Understanding the dimensionality of humor representations would:
- Provide insights into how LLMs process complex semantic concepts
- Inform efficient fine-tuning strategies for humor-related tasks
- Contribute to the broader understanding of linear representations in LLMs

## Hypothesis Decomposition

### H1: Humor is linearly separable
- A linear probe on hidden representations should classify humor vs. non-humor with high accuracy (>80%)
- Different probing methods (mean difference, logistic regression, PCA) should find similar directions

### H2: The rank is low
- A small number of principal components should capture most variance between humor/non-humor
- Performance should plateau quickly as we add more dimensions
- Effective rank should be <10 dimensions (based on LoRA findings)

### H3: Layer specificity exists
- Middle-to-late layers should encode humor better than early layers (similar to sentiment)
- The best layer should significantly outperform random baseline

### H4: The representation generalizes
- Probes trained on one subset should transfer to held-out data
- Mean difference direction should align with logistic regression weights

## Proposed Methodology

### Approach

1. **Extract activations** from a pre-trained LLM (GPT-2-small for computational efficiency)
2. **Train linear probes** at each layer to identify where humor is encoded
3. **Analyze rank** using PCA on class-separated activations
4. **Validate** with multiple probing methods and cross-validation

### Why This Model?
- GPT-2-small (124M params) is well-understood in interpretability literature
- Supported by TransformerLens for easy activation extraction
- Sufficient capacity to encode semantic features
- Computationally feasible for multiple experiments

### Experimental Steps

#### Step 1: Data Preparation
- Load ColBERT Humor dataset (200k samples)
- Split: 80% train, 10% val, 10% test (160k/20k/20k)
- Tokenize with GPT-2 tokenizer
- Extract final token position for each text (following sentiment paper methodology)

#### Step 2: Activation Extraction
- Use TransformerLens HookedTransformer
- Extract residual stream activations at all layers (12 layers for GPT-2-small)
- Focus on final token position (where information aggregates)
- Store activations for train/val/test splits

#### Step 3: Linear Probing (per layer)
- **Mean Difference**: μ_humor - μ_non_humor
- **Logistic Regression**: Train on train set, evaluate on test set
- Compare probe accuracy across layers

#### Step 4: Rank Analysis
- Compute class means for humor and non-humor
- Perform PCA on pooled activations
- Analyze explained variance vs. number of components
- Determine effective rank (components needed for 90%, 95%, 99% variance)
- Test classification accuracy vs. number of PCA components

#### Step 5: Validation
- Cross-validate probing results (5-fold)
- Compare direction vectors from different methods (cosine similarity)
- Test generalization to held-out test set

### Baselines

1. **Random direction**: Classify using random unit vector
2. **Random guess**: 50% accuracy for balanced dataset
3. **Full embedding**: Use all 768 dimensions without reduction

### Evaluation Metrics

**Classification**:
- Accuracy (primary, dataset is balanced)
- F1-score (for robustness)
- AUC-ROC (for confidence analysis)

**Rank Analysis**:
- Explained variance ratio per component
- Cumulative explained variance
- Accuracy vs. rank curve
- Effective rank at 90%/95%/99% variance

**Direction Similarity**:
- Cosine similarity between different probing method directions

### Statistical Analysis Plan

- 95% confidence intervals via bootstrap (1000 samples)
- Paired t-test for comparing methods (significance: p < 0.05)
- Effect size (Cohen's d) for practical significance
- Bonferroni correction for multiple comparisons across layers

## Expected Outcomes

### Supporting the Hypothesis
- Linear probe accuracy >80% at best layer
- Effective rank <10 for 90% explained variance
- Clear accuracy plateau as dimensions increase
- High cosine similarity (>0.7) between different probing methods

### Refuting the Hypothesis
- Linear probe accuracy near random (50%)
- High rank required (>100 dimensions)
- No clear structure in PCA decomposition
- Low correlation between different probing methods

### Interesting Alternate Outcomes
- Moderate rank (10-50): Humor is more complex than sentiment but still structured
- Layer-dependent rank: Early layers low-rank, later layers higher
- Non-linear separability: Good SVM but poor linear probe

## Timeline and Milestones

| Phase | Estimated Time | Deliverable |
|-------|---------------|-------------|
| Environment Setup | 10 min | Working env with dependencies |
| Data Preparation | 15 min | Loaded and split dataset |
| Activation Extraction | 30 min | Cached activations for all samples |
| Linear Probing | 20 min | Probe accuracy per layer |
| Rank Analysis | 20 min | PCA results and plots |
| Validation | 15 min | Cross-validation and similarity |
| Documentation | 20 min | REPORT.md complete |

## Potential Challenges

1. **Memory constraints**: 200k samples × 768 dims × 12 layers is large
   - *Mitigation*: Use subset (10-20k samples), process in batches

2. **Computational time**: Inference on 200k samples takes time
   - *Mitigation*: Start with smaller subset, scale up if time permits

3. **Model download**: TransformerLens needs to download GPT-2
   - *Mitigation*: Standard HuggingFace download, should work

4. **Dataset format**: May need format conversion
   - *Mitigation*: Check format early, adapt loading code

## Success Criteria

**Primary Success** (must achieve):
- Successfully extract activations from GPT-2
- Train and evaluate linear probes at each layer
- Generate rank vs. accuracy analysis
- Document findings in REPORT.md

**Secondary Success** (nice to have):
- Cross-validation of all results
- Multiple probing methods compared
- Clear visualizations of rank structure
- Causal validation via activation patching

## Resource Requirements

- CPU sufficient for GPT-2-small inference (GPU preferred but not required)
- ~8GB RAM for activation caching
- ~30 min total runtime for core experiments
- Python packages: torch, transformers, transformer-lens, scikit-learn, matplotlib
