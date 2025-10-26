# Rigorous Analysis of Temporal Scale Detection Results

**Date**: October 26, 2025
**Status**: Critical Verification Phase

---

## 🔍 Executive Summary

After fixing a critical methodological flaw (data leakage in probe training), we observe:
- **Test accuracy: 99%** at Layer 9 (was 84% with flawed methodology)
- **Control (keyword ablated): ~50%** (was incorrectly showing 100%)

However, **we must verify these results carefully** before drawing conclusions.

---

## 1. Methodological Fix: What Changed

### Original Methodology (FLAWED):
```python
# For "test" results:
cv_scores = cross_val_score(probe, X_test, y_test, cv=5)
# Problem: Uses test data for BOTH training and testing through CV folds
```

**Issue**: The probe sees test data during cross-validation, creating data leakage.

### Corrected Methodology:
```python
# Train on training set only
probe.fit(X_train, y_train)

# Evaluate on HELD-OUT test set (probe has never seen this)
test_accuracy = probe.score(X_test, y_test)
```

**Fix**: Clean separation - probe trained only on train set, evaluated on unseen test set.

---

## 2. Complete Results Table

| Layer | Train CV (mean±std) | Val Acc | Test Acc | Control Acc | Δ (Test-Control) |
|-------|---------------------|---------|----------|-------------|------------------|
| 0 | 68.0% ± 5.6% | 78.0% | 80.0% | 49.0% | +31.0pp |
| 1 | 74.8% ± 10.1% | 78.0% | 82.0% | 51.0% | +31.0pp |
| 2 | 82.0% ± 6.6% | 84.0% | 84.0% | 51.0% | +33.0pp |
| 3 | 83.8% ± 11.9% | 83.0% | 89.0% | 49.0% | +40.0pp |
| 4 | 87.5% ± 6.6% | 89.0% | 91.0% | 44.0% | +47.0pp |
| 5 | 90.0% ± 4.7% | 96.0% | 93.0% | 47.0% | +46.0pp |
| 6 | 91.0% ± 7.1% | 97.0% | 96.0% | 47.0% | +49.0pp |
| 7 | 91.5% ± 7.1% | 98.0% | 95.0% | 47.0% | +48.0pp |
| 8 | 92.5% ± 6.4% | 100.0% | 95.0% | 49.0% | +46.0pp |
| **9** | **90.0% ± 4.2%** | **97.0%** | **99.0%** | **45.0%** | **+54.0pp** ⭐ |
| 10 | 86.0% ± 3.2% | 94.0% | 96.0% | 41.0% | +55.0pp |
| 11 | 85.3% ± 4.2% | 96.0% | 96.0% | 26.0% | +70.0pp |

### Key Observations:

1. **Test accuracy peaks at Layer 9: 99%** (49/50 correct)
2. **Control accuracy ~50%** (chance level) across layers 0-9
3. **Control drops to 26-41%** in layers 10-11 (below chance - inverted predictions?)
4. **Validation accuracy often EXCEEDS test** (Layer 8: 100% val, 95% test)

---

## 3. Red Flags & Questions to Investigate

### 🚨 Red Flag #1: Val > Test Performance

**Observation**: Validation accuracy (100% at Layer 8) exceeds test accuracy (95%)

**Possible explanations**:
1. Val set is easier than test set (different distributions?)
2. Lucky split - val set happens to be more separable
3. Small sample size (100 samples each) → high variance
4. Overfitting to validation set (though we don't tune on it)

**Need to verify**:
- Check val/test set distributions
- Examine specific examples where val succeeds but test fails

### 🚨 Red Flag #2: Near-Perfect 99% Test Accuracy

**Observation**: 99% test accuracy (only 1 error out of 100 samples)

**Possible explanations**:
1. ✅ Task is genuinely easy for the model
2. ⚠️ Test set is not representative
3. ⚠️ Temporal keywords are extremely salient features
4. ⚠️ Some confound we haven't identified

**Need to verify**:
- Manual inspection of the 1 misclassified example
- Check if test set has systematic biases
- Compare keyword saliency across datasets

### 🚨 Red Flag #3: Control Below Chance in Late Layers

**Observation**: Layers 10-11 achieve 26-41% on control (worse than 50% chance)

**Possible explanations**:
1. ✅ Probe learned to rely on temporal keywords, gets confused when absent
2. ✅ Systematic bias in how keywords were ablated
3. ⚠️ Layer 11 at 26% suggests probe is **inverting** predictions
4. ⚠️ Could indicate the ablation creates misleading patterns

**Need to verify**:
- Examine confusion matrices for control predictions
- Check if ablation methodology is consistent
- Understand why late layers perform worse than early layers on control

---

## 4. Dataset Examination

Let me load and examine the actual prompts to understand what we're testing:

### Sample Size Verification

```
Training set: 400 samples (200 pairs)
Validation set: 100 samples (50 pairs)
Test set: 100 samples (50 pairs)
Control (ablated): 100 samples (50 pairs)
```

**Question**: Are these drawn from the same 300-pair pool or different?

### Need to Examine:

1. **Test set specific examples**:
   - What is the 1 misclassified example at Layer 9?
   - Are there systematic patterns in errors?

2. **Control set examples**:
   - How exactly were keywords ablated?
   - Is there information leakage in the ablation?

3. **Distribution checks**:
   - Domain balance across train/val/test
   - Temporal scale distribution
   - Prompt length distributions

---

## 5. Comparison to Prior Work

### Linear Probing Baseline Expectations

Based on prior interpretability research using linear probes:

#### Similar Studies & Expected Performance:

**1. Sentiment Classification (Radford et al., 2017 - OpenAI Sentiment Neuron)**
- Task: Binary sentiment (positive/negative)
- Method: Linear probe on LSTM states
- Result: ~92% accuracy on movie reviews
- Comparison: Our 99% is higher but task may be easier

**2. Subject-Verb Agreement (Lakretz et al., 2019)**
- Task: Grammatical number agreement
- Method: Linear probes on LSTM/Transformer layers
- Result: ~85-90% accuracy on synthetic data
- Comparison: Similar to our mid-layer performance

**3. Part-of-Speech Tagging (Tenney et al., 2019 - BERT Probing)**
- Task: POS tags across layers
- Method: Linear probes on BERT
- Result: 85-95% accuracy depending on layer/tag
- Comparison: Our performance is in typical range

**4. Negation Detection (Ettinger, 2020)**
- Task: Detecting negation in context
- Method: Probes on BERT layers
- Result: 75-85% accuracy
- Comparison: Our task shows higher accuracy

**5. Factual Knowledge (Petroni et al., 2019 - LAMA)**
- Task: Fact retrieval (e.g., "Paris is the capital of ___")
- Method: Probing LM predictions
- Result: 30-60% accuracy depending on fact type
- Comparison: Our task is much easier

### Key Insights from Literature:

1. **90-95% is typical for "easy" binary classification tasks**
   - When features are salient and separable
   - Linear probes can achieve very high accuracy
   - Examples: sentiment, simple grammar rules

2. **99% accuracy is UNUSUAL but not impossible**
   - Suggests highly salient features
   - Could indicate task is "too easy" (lexical shortcuts)
   - Need strong controls to verify semantic understanding

3. **Control performance at chance is EXPECTED**
   - If probes rely on surface features
   - Ablating those features should drop to ~50%
   - This is actually evidence AGAINST deep semantic encoding

### Red Flag from Literature:

**Hewitt & Liang (2019): "Designing and Interpreting Probes with Control Tasks"**

Key finding: High probe accuracy doesn't always mean the representation encodes the property.

**Warning signs**:
- Probe can memorize based on small dataset
- Hyperparameters matter more than representation quality
- Need control tasks that are equally "complex" but random

**Our situation**:
- ✅ We have control tasks (keyword ablation)
- ⚠️ Control drops to chance → suggests lexical features dominate
- ❓ Is 99% due to representation quality or task simplicity?

---

## 6. Critical Questions to Answer

### Q1: What is the 1 error at Layer 9?

**Need**: Examine the specific example that was misclassified

**Why**: Could reveal systematic failure mode or be random noise

### Q2: How salient are temporal keywords?

**Hypothesis**: Keywords like "1 week" vs "20 years" are EXTREMELY salient

**Test**:
- Measure token-level saliency (attention weights)
- Compare saliency to other tokens
- Check if ANY other features contribute

### Q3: Is the control ablation methodology correct?

**Current approach**: Replace temporal keywords with placeholders

**Questions**:
- What exactly is the replacement? "[TIME]"? Blank?
- Does replacement preserve token count?
- Could replacement tokens themselves be informative?

### Q4: Can we create harder controls?

**Proposals**:
1. **Swap control**: Use wrong temporal keyword
   - "1 week" → "20 years" but keep task
   - Tests if probe uses keyword or context

2. **Position control**: Extract activations BEFORE temporal keyword
   - Tests if model anticipates temporal scale

3. **Semantic control**: Complex short-term vs simple long-term
   - Tests task complexity confound

### Q5: How does this compare to human performance?

**Need**: Estimate human accuracy on this task

**Expectation**: Likely ~100% for humans (trivial task)

**Implication**: If humans are perfect and model is 99%, maybe this is just an easy task

---

## 7. Token-Level Analysis Needed

### Hypothesis: Model is just detecting tokens

**Evidence FOR**:
- Control (ablated) drops to ~50% (chance)
- Test accuracy is very high (99%)
- Task involves explicit temporal keywords

**Evidence AGAINST**:
- Would need to see token-level saliency analysis
- Check if model uses context beyond keywords

### What to measure:

1. **Attention weights** on temporal tokens vs other tokens
2. **Gradient-based attribution** - which tokens matter most?
3. **Token swap experiments** - swap temporal keywords, see if classification changes

---

## 8. Dataset Balance & Composition Check

### Need to verify:

```python
# Domain distribution
# Temporal scale distribution
# Prompt length distribution
# Keyword variation
```

**Questions**:
- Are all domains equally represented in train/val/test?
- Is there variety in temporal expressions?
- Could length be a confound?

---

## 9. Conservative Interpretation (Current Evidence)

### What we CAN claim:

1. ✅ GPT-2 encodes information sufficient to distinguish temporal scale prompts with 99% accuracy
2. ✅ This information emerges progressively through layers (68% → 99%)
3. ✅ Information is concentrated in middle-to-late layers (peak at Layer 9)
4. ✅ Linear probes can extract this information reliably

### What we CANNOT yet claim:

1. ❌ Information is "semantic" vs "lexical" (control suggests lexical)
2. ❌ Model "understands" temporal scale (vs pattern matching)
3. ❌ Would generalize beyond explicit temporal keywords
4. ❌ Represents abstract temporal reasoning

### What we SHOULD claim (conservative):

> "Linear probes trained on GPT-2 activations achieve 99% accuracy in classifying
> prompt pairs that differ only in temporal scale keywords (e.g., '1 week' vs '20 years').
> This performance drops to chance level (~50%) when temporal keywords are ablated,
> suggesting the model primarily relies on lexical features rather than deeper semantic
> understanding of temporal scale."

---

## 10. Recommended Next Steps

### Immediate (Required before publication):

1. **Manual inspection**:
   - All misclassified examples
   - Random sample of correct classifications
   - Verify data quality

2. **Statistical tests**:
   - Significance tests for layer differences
   - Effect sizes (not just accuracy)
   - Bootstrap confidence intervals

3. **Ablation study**:
   - Exact methodology documentation
   - Check for artifacts in ablation
   - Verify replacement strategy

4. **Token-level analysis**:
   - Attention weights on temporal keywords
   - Gradient attribution scores
   - Confirm keyword-dependence

### Short-term (Strengthen claims):

5. **Enhanced controls**:
   - Keyword swap experiments
   - Position-matched extraction
   - Task complexity controls

6. **Cross-validation**:
   - Different train/test splits
   - Bootstrap stability analysis
   - Domain-specific analysis

### Long-term (Full paper):

7. **Causal analysis**:
   - Activation patching
   - Attention head ablation
   - Circuit identification

8. **Cross-model validation**:
   - GPT-2 medium/large
   - Other architectures
   - Scale dependence

---

## 11. Preliminary Conclusion (Requires Verification)

The corrected methodology reveals:

**Strong evidence FOR**:
- Robust temporal scale classification (99% test accuracy)
- Hierarchical information accumulation
- Reliable probe performance

**Strong evidence AGAINST**:
- Deep semantic understanding (control at chance)
- Keyword-independent reasoning
- Abstract temporal representation

**Most likely explanation**:
GPT-2 learns to strongly associate certain tokens (temporal keywords) with distinct
activation patterns, enabling near-perfect linear classification. However, this appears
to be lexical pattern matching rather than compositional temporal reasoning.

**This is still valuable research** - it shows:
1. How strongly temporal keywords are encoded
2. Where in the network this encoding happens
3. The limitations of lexical vs semantic encoding

**Publication-worthy IF**:
- We verify data quality
- We run additional controls
- We frame claims conservatively
- We position as "temporal keyword detection" not "temporal understanding"

---

## CRITICAL: Data Verification Checklist

Before proceeding, we MUST verify:

- [ ] Examine train/val/test split methodology
- [ ] Inspect the 1 misclassified example at Layer 9
- [ ] Verify control ablation methodology
- [ ] Check for data contamination between splits
- [ ] Analyze domain/length distributions
- [ ] Manual review of random sample (n=20)
- [ ] Verify label correctness
- [ ] Check for duplicate prompts

**Status**: VERIFICATION IN PROGRESS

---

**Last Updated**: October 26, 2025
**Confidence Level**: Medium (awaiting verification)
**Recommendation**: Proceed with detailed data examination before finalizing claims
