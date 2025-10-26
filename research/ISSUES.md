# Open Research Issues

## Issue #1: Verify Temporal Steering is Not Confounded by Style

**Status**: 🔴 Open
**Priority**: High
**Created**: 2025-10-26

### Problem

Initial confound analysis (2×2 design, 40 prompts) shows promising PCA separation between temporal and style dimensions. However, several concerns remain:

1. **Style-Temporal Coupling**: Formal language may naturally imply longer planning horizons
   - Formal prompts use words like "institutional frameworks", "multi-generational"
   - Casual prompts use "now", "asap", "quick fixes"
   - This creates potential confound: is temporal steering just detecting formality?

2. **Small Sample Size**: Only 10 prompts per cell (40 total)
   - Insufficient for robust statistical conclusions
   - High variance in PCA projections
   - Need 50-100+ prompts per cell for confidence

3. **Prompt Length Confound**: Casual prompts are systematically shorter
   - Casual: ~10 words average
   - Formal: ~15 words average
   - Could temporal effect actually be a length effect?

4. **Manual Construction Bias**: Prompts hand-crafted, not systematically generated
   - Researcher expectations may leak into prompt design
   - Need programmatic generation or crowdsourcing

### Current Evidence

**PCA Results** (`research/results/pca_temporal_style.png`):
- PC1 appears to separate temporal (18.1% variance)
- PC2 appears to separate style (13.6% variance)
- Visual separation looks clean, but quantitative metrics show overlap

**Separation Metrics**:
```
Temporal separation: PC1=14.5, PC2=14.6 (ratio 0.99x) ⚠️
Style separation: PC1=14.9, PC2=8.9 (ratio 0.59x)
```

Interpretation: Both factors present in both PCs, though visual inspection suggests cleaner separation.

### Proposed Solutions

#### Solution 1: Expand Dataset (Priority: High)

**Target**: 100 prompts per cell (400 total)

**Generation Strategy**:
1. **Systematic variation**: For each topic, create 4 orthogonal variants
   - Same semantic content
   - Vary temporal scope independently of style
   - Vary style independently of temporal scope

2. **Example template**:
   ```
   Topic: Transportation

   Immediate + Casual: "Traffic's terrible, what quick fix can we try?"
   Immediate + Formal: "Current traffic conditions require intervention. Recommendations?"

   Long-term + Casual: "What transport will kids use in 2075?"
   Long-term + Formal: "What transportation infrastructure ensures long-term sustainability?"
   ```

3. **Control for confounds**:
   - Match word count across conditions (±2 words)
   - Avoid temporal keywords in style manipulation
   - Avoid style keywords in temporal manipulation
   - Use same verbs/nouns across conditions

4. **Validation**:
   - Human raters verify temporal scope (blind to condition)
   - Human raters verify style (blind to condition)
   - Inter-rater reliability > 0.8

#### Solution 2: Behavioral Validation (Priority: High)

**Test**: Do deconfounded vectors actually affect temporal scope?

1. **Generation test**:
   - Prompt: "How should we address climate change?" (neutral)
   - Generate with:
     - Original temporal vectors (±1.0)
     - Deconfounded temporal vectors (±1.0)
     - No steering (baseline)
   - 10 generations per condition, temperature=0.7

2. **Human evaluation** (blind):
   - Raters judge temporal horizon: immediate/short/medium/long
   - Raters judge formality: casual/neutral/formal
   - Check if deconfounded vectors shift temporal WITHOUT shifting style

3. **Automatic metrics**:
   - Count temporal markers (now, today, future, decades, etc.)
   - Measure planning horizon (explicit time references)
   - Measure formality (lexical complexity, sentence structure)

4. **Acceptance criteria**:
   - Deconfounded vectors significantly shift temporal markers (p < 0.05)
   - Deconfounded vectors do NOT significantly shift formality (p > 0.10)
   - Original vs deconfounded vectors have similar temporal effect sizes

#### Solution 3: Multi-Model Validation (Priority: Medium)

**Test on multiple models**:
- GPT-2 (124M) ✓ [done]
- GPT-2-large (774M)
- LLaMA-2-7B
- Mistral-7B

**Expected results**:
- If temporal-style separation is real: should replicate across models
- If it's GPT-2 specific: confound or model artifact

#### Solution 4: Regression Analysis (Priority: Medium)

**Quantify factor contributions**:

1. **Linear regression on PC1**:
   ```python
   PC1 ~ temporal + style + temporal×style + length + complexity
   ```
   - Measure unique variance explained by temporal
   - Measure unique variance explained by style
   - Check for interaction effects

2. **Expected if not confounded**:
   - Temporal predicts PC1: β > 0.5, p < 0.01
   - Style predicts PC1: β < 0.2, p > 0.05
   - Interaction: β < 0.1

3. **Alternative: Factor Analysis**:
   - Fit 2-factor model to activations
   - Check if factors align with temporal/style
   - Measure factor independence (correlation < 0.3)

### Acceptance Criteria

Issue can be closed when ALL of the following are met:

- [ ] **Dataset expansion**: 100+ prompts per cell, validated by humans
- [ ] **Behavioral test**: Deconfounded vectors shift temporal scope (p < 0.05)
- [ ] **Style control**: Deconfounded vectors do NOT shift formality (p > 0.10)
- [ ] **Multi-model**: Results replicate on ≥2 models (GPT-2-large, LLaMA-2)
- [ ] **Statistical test**: Regression shows temporal uniquely predicts PC1
- [ ] **Documentation**: Update CONFOUND_ANALYSIS_RESULTS.md with findings
- [ ] **Publication ready**: Results robust enough for paper/blog post

### Related Files

- `research/datasets/confound_experiment.json` - Current 40-prompt dataset
- `research/experiments/confound_analysis.py` - Analysis code
- `research/results/pca_temporal_style.png` - PCA visualization
- `research/results/CONFOUND_ANALYSIS_RESULTS.md` - Current findings
- `steering_vectors/temporal_scope_deconfounded.json` - Deconfounded vectors

### Next Steps

1. **Immediate** (this week):
   - [ ] Create prompt generation script with length/complexity controls
   - [ ] Generate 400-prompt dataset (100 per cell)
   - [ ] Recruit human raters for validation (or use crowdsourcing)

2. **Short-term** (next 2 weeks):
   - [ ] Run behavioral evaluation (generation + human rating)
   - [ ] Implement regression analysis
   - [ ] Test on GPT-2-large

3. **Medium-term** (next month):
   - [ ] Port to LLaMA-2-7B using model adapter
   - [ ] Write up findings
   - [ ] Update steering vector files with validated versions

### Discussion

**Why this matters**:
- If temporal steering is just a style confound, the technique is not useful
- Validation is critical before publishing or productionizing
- Other researchers need to trust these vectors work as advertised

**What would invalidate the approach**:
- Behavioral test shows deconfounded vectors only affect style
- Multi-model test shows effect doesn't replicate
- Larger dataset shows PC1 is primarily style, not temporal

**What would validate the approach**:
- Clean behavioral separation (temporal shifts without style shifts)
- Replication across models
- Regression shows temporal is unique predictor of PC1

---

**Assigned to**: TBD
**Labels**: research, validation, high-priority
**Milestone**: v1.0 validation

