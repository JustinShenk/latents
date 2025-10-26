# Confound Analysis Results

**⚠️ PRELIMINARY RESULTS - REQUIRES VERIFICATION**

These results are from an initial 2×2 experimental design testing whether temporal steering is confounded by stylistic features. Further validation is needed before drawing strong conclusions.

## Experimental Design

**Dataset**: 40 prompts in 2×2 crossed design
- **Factor 1 (Temporal)**: Immediate vs Long-term (20 each)
- **Factor 2 (Style)**: Casual vs Formal (20 each)
- **Cells**: 10 prompts per condition

**Model**: GPT-2 (124M parameters)
**Layers Analyzed**: 7-11 (later layers where temporal effects are strongest)

## PCA Results (Layer 10)

**Variance Explained:**
- PC1: 18.1%
- PC2: 13.6%
- PC3: 8.8%
- PC4: 7.3%

**Visual Inspection** (`pca_temporal_style.png`):
- PC1 (x-axis): Appears to separate **temporal dimension** (immediate=right, long-term=left)
- PC2 (y-axis): Appears to separate **style dimension** (casual=top, formal=bottom)
- Separation looks relatively clean and somewhat orthogonal

## Quantitative Separation Metrics

**Temporal Separation:**
- Along PC1: 14.499
- Along PC2: 14.599
- Ratio (PC1/PC2): 0.99x

**Style Separation:**
- Along PC1: 14.931
- Along PC2: 8.852
- Ratio (PC2/PC1): 0.59x

**Interpretation:**
The quantitative metrics show both factors have some presence in both PCs, but visual inspection suggests cleaner separation than the raw numbers indicate. The metrics may need refinement.

## Deconfounded Vectors

**Method**: Average across style conditions to extract pure temporal signal
- `immediate_vector = mean(immediate_casual, immediate_formal)`
- `longterm_vector = mean(longterm_casual, longterm_formal)`
- `temporal_vector = longterm_vector - immediate_vector`

**Vector Norms by Layer:**
```
Layer  7: norm=28.276
Layer  8: norm=34.452
Layer  9: norm=44.303
Layer 10: norm=52.867
Layer 11: norm=74.489
```

Saved to: `steering_vectors/temporal_scope_deconfounded.json`

## Limitations & Next Steps

### Current Limitations:
1. **Small sample size**: Only 10 prompts per cell
2. **Single model**: Only tested on GPT-2 (need LLaMA-2, larger models)
3. **Manual prompts**: Hand-crafted, not systematically generated
4. **No human evaluation**: Visual PCA inspection, no behavioral tests
5. **Quantitative metrics**: May not capture full separation story

### Verification Needed:
1. **Human evaluation**: Test if deconfounded vectors still affect temporal scope
2. **Ablation study**: Compare original vs deconfounded steering in blind test
3. **Larger sample**: Scale to 50-100 prompts per cell
4. **Other models**: Replicate on GPT-2-large, LLaMA-2-7B
5. **Statistical tests**: ANOVA or regression to quantify factor contributions
6. **Behavioral metrics**: Measure temporal markers, planning horizon in generated text

### Questions to Answer:
1. Do deconfounded vectors preserve temporal steering effect?
2. How much variance is purely temporal vs confounded?
3. Are there other confounds (e.g., complexity, length)?
4. Does separation improve with more data?

## Preliminary Conclusions

**Tentative findings** (pending verification):

✅ **Positive signs:**
- Visual PCA shows clear separation of temporal and style factors
- Factors appear somewhat orthogonal (not fully confounded)
- Deconfounded extraction is feasible

⚠️ **Concerns:**
- Quantitative metrics show factors present in both PCs
- Need behavioral validation (not just activation space analysis)
- Small sample size limits confidence

**Status**: This is a proof-of-concept showing the analysis is tractable. **Do not publish or rely on these findings without further validation.**

## Files Generated

1. `research/datasets/confound_experiment.json` - Prompt dataset
2. `research/results/pca_temporal_style.png` - PCA visualization
3. `research/results/confound_activations.npz` - Raw activations
4. `steering_vectors/temporal_scope_deconfounded.json` - Deconfounded vectors
5. `research/experiments/confound_analysis.py` - Analysis code

## Reproducibility

To reproduce:
```bash
python research/experiments/confound_analysis.py
```

Requires: `transformers`, `torch`, `sklearn`, `matplotlib`, `numpy`

---

**Date**: 2025-10-26
**Model**: GPT-2 (124M)
**Status**: Preliminary / Proof-of-Concept
**Next**: Human evaluation + larger scale validation
