# Rigorous Experimental Design for Temporal Reasoning

**Based on critical feedback - implementing proper confound controls**

---

## 1. Controlled Confound Management

### Principle: Minimal Substitution Design

**Goal**: Same everything except temporal scope markers

**Template**:
```
"What should [AGENT] prioritize for [TEMPORAL_MARKER] [GOAL]?"
```

**Controlled pairs**:
```
Immediate: "What should city planners prioritize for next month's transportation needs?"
Long-term:  "What should city planners prioritize for next generation's transportation needs?"

Immediate: "What should investors consider for this quarter's portfolio decisions?"
Long-term:  "What should investors consider for their children's portfolio decisions?"

Immediate: "What actions will address today's housing crisis?"
Long-term:  "What actions will address future generations' housing needs?"
```

### Why This Works

**Controls for**:
- ✅ Topic (identical)
- ✅ Vocabulary (95% overlap)
- ✅ Sentence structure (identical)
- ✅ Complexity (same)
- ✅ Semantic content (same core question)

**Varies only**:
- Temporal scope marker ("next month" vs "next generation")
- No explicit numbers!

### Dataset Structure

**50 controlled pairs**:
- Same sentence template
- Minimal word substitution
- Temporal marker is ONLY difference
- No keyword detection possible

---

## 2. Performance-Based Validation

### Principle: Temporal Scope Must Affect Correct Answer

**Current problem**: Classification task has no "ground truth" - we just label them ourselves

**Better approach**: Use tasks where temporal horizon demonstrably affects the correct action

### Example 1: Financial Decision

```
Context: "Interest rates are at historic lows (1%) but economists predict they'll rise to 5% within two years."

Question A: "Should I refinance my mortgage now or wait?"
Correct answer: NOW (rates rising)

Question B: "Should I lock in a 30-year fixed rate or use adjustable?"
Correct answer: FIXED (long-term protection)

Question C: "Should I pay down debt aggressively or invest in stocks?"
Depends on: Time horizon for investing
```

**Test**: Can model give DIFFERENT answers based on explicit temporal framing?

```
Prompt 1: "Given my retirement is in 30 years, should I [...]"
Expected: Long-term strategy

Prompt 2: "Given I need this money next month, should I [...]"
Expected: Short-term strategy
```

### Example 2: Climate Policy

```
Context: "A coal plant can be shut down immediately (causing job losses) or phased out over 20 years (allowing transition)."

Immediate question: "What should policymakers do to reduce emissions TODAY?"
Expected answer: Phase-out (balances immediate action with transition time)

Long-term question: "What should policymakers do to ensure zero-carbon energy for future generations?"
Expected answer: Immediate shutdown + massive renewable investment (long-term thinking)
```

### Performance Validation Protocol

1. **Create 20 questions** with temporal-dependent correct answers
2. **Extract model completions** for immediate vs long-term framings
3. **Verify model gives DIFFERENT answers**
4. **If model doesn't change answers** → probe is detecting noise, not meaningful temporal understanding

---

## 3. Adversarial Testing FIRST

### Principle: Swap Temporal Keywords Between Contexts

**Test cases**:

```
Natural match:
"What can we do RIGHT NOW to prevent this forest fire from spreading?"
→ Expected classification: IMMEDIATE ✓

Adversarial swap:
"What foundations should we build for future generations to prevent this forest fire from spreading?"
→ Contains long-term keyword but semantically nonsensical
→ If probe says LONG-TERM: it's just detecting keywords
→ If probe says IMMEDIATE: it might understand context
```

### Adversarial Dataset (50 pairs)

**Type 1: Semantic contradiction**
```
Long keyword + Immediate context:
"What legacy will we leave for posterity by fixing this urgent crisis in the next hour?"

Expected: Probe should be confused (both signals present)
Ideal: Probe classifies as IMMEDIATE (semantic context wins)
Reality check: Probe probably follows keyword
```

**Type 2: Irrelevant keyword**
```
"In the long run, what should we do immediately?"
→ Both temporal markers present
→ Tests if probe integrates context or just pattern matches
```

**Type 3: Opposite framing**
```
Take a LONG-TERM task, frame it with IMMEDIATE language:
"We need to transform our entire civilization RIGHT NOW"
→ Semantically confused
→ Tests robustness
```

### Adversarial Testing Protocol

**Run this BEFORE main experiment**:
1. Generate 50 adversarial examples
2. Classify with existing probes
3. If accuracy > 60%: Probe is using semantic reasoning
4. If accuracy ~50%: Probe is confused (good - shows it needs both signals)
5. If accuracy follows keywords: Probe is doing keyword detection

**Decision rule**:
- If adversarial test shows keyword-only detection → STOP, redesign
- If adversarial test shows semantic integration → PROCEED with full experiment

---

## 4. Mechanistic Hypothesis First

### Principle: Predict Circuits Before Testing

**Core question**: What neural circuits handle temporal scope?

### Hypothesis 1: Temporal Scope = Dedicated Circuit

**Prediction**:
- Specific attention heads encode temporal markers
- Located in middle-late layers (6-9)
- Independent from complexity/abstraction circuits

**Test**:
- Attention head ablation
- Should find 2-4 heads critical for temporal detection
- Ablating these heads should drop accuracy dramatically

**Alternative hypothesis**: If ablating individual heads doesn't matter, temporal scope might be distributed or use existing circuits

### Hypothesis 2: Temporal Scope Reuses Abstraction Circuit

**Prediction**:
- Same heads that encode abstract vs concrete
- Or same heads that encode scope/scale generally
- Temporal is just one dimension of "scope"

**Test**:
- Compare with abstraction detection task
- If same heads light up → temporal reuses abstraction
- If different heads → temporal has dedicated circuit

### Hypothesis 3: Temporal Scope = Keyword Detection Only

**Prediction**:
- Early layer token embeddings sufficient
- No need for deep processing
- Ablating late layers doesn't hurt performance

**Test**:
- Probe early layers (0-3) with keyword-based features
- If accuracy similar to late layers → just keyword detection
- If late layers much better → something more sophisticated

### Circuit Isolation Protocol

**Step 1: Attention Pattern Analysis**
```python
# For each temporal prompt pair, examine:
# - Which heads attend to temporal markers?
# - Do they attend differently for short vs long?
# - Are attention patterns consistent across examples?
```

**Step 2: Causal Intervention (Activation Patching)**
```python
# Replace activations from short prompt with long prompt activations
# At each layer, each head
# Measure: Does classification flip?
# If yes → that component is causally important
```

**Step 3: Head Ablation**
```python
# Zero out specific attention heads
# Measure accuracy drop
# Identify 3-5 most critical heads
```

**Step 4: Minimal Circuit**
```python
# Can we reconstruct temporal detection with just the critical heads?
# Build "minimal temporal circuit"
# Test if it generalizes to implicit temporal scope
```

### Predicted Circuit (To Be Validated)

**Speculative architecture**:
1. **Layer 2-3**: Token-level temporal marker detection
   - Heads recognize "year", "decade", "week" tokens
   - Build local temporal context

2. **Layer 5-6**: Integration of temporal markers with task context
   - Heads combine temporal scope with task complexity
   - Build coherent temporal frame

3. **Layer 8-9**: Abstract temporal scope representation
   - Heads encode "immediate" vs "long-term" as high-level feature
   - Independent of specific keywords

**Validation**:
- If this circuit exists → proves compositional temporal understanding
- If doesn't exist → back to keyword detection hypothesis

---

## 5. Proper Experimental Sequence

### Phase 0: Adversarial Pre-Test (1 day)

1. Generate 50 adversarial keyword-swap examples
2. Test with existing probes
3. **DECISION GATE**:
   - If keyword-only detection → redesign
   - If semantic integration → proceed

### Phase 1: Controlled Minimal-Substitution Dataset (3 days)

1. Generate 100 controlled pairs (same structure, minimal substitution)
2. Manual validation (no confounds)
3. Extract activations
4. Train probes with proper train/test split
5. **Baseline performance** on controlled dataset

### Phase 2: Performance-Based Validation (3 days)

1. Create 20 temporal-dependent decision tasks
2. Extract model completions
3. Verify model behavior changes with temporal framing
4. **Ground truth**: Probe should correlate with model's actual behavior

### Phase 3: Circuit Analysis (1 week)

1. Attention pattern analysis
2. Activation patching
3. Head ablation
4. Identify minimal circuit
5. **Mechanistic understanding** of temporal processing

### Phase 4: Implicit Temporal Test (2 days)

1. Test circuit on implicit temporal prompts
2. If circuit generalizes → true semantic understanding
3. If doesn't generalize → limited to explicit framing

---

## 6. Success Criteria (Before Starting)

### Strong Evidence for Temporal Understanding

- ✅ Controlled dataset: >70% accuracy
- ✅ Adversarial test: <60% (probe is confused by conflicts)
- ✅ Performance validation: Model behavior correlates with probe
- ✅ Circuit analysis: 3-5 critical heads identified
- ✅ Implicit temporal: >60% accuracy

### Evidence for Keyword Detection Only

- ⚠️ Controlled dataset: >90% accuracy
- ⚠️ Adversarial test: >80% (follows keywords)
- ⚠️ Performance validation: No correlation with model behavior
- ⚠️ Circuit analysis: No specific heads (distributed)
- ⚠️ Implicit temporal: ~50% (chance)

### Null Result

- ❌ Controlled dataset: <60% accuracy
- ❌ Task is too hard even with controlled prompts
- ❌ GPT-2 doesn't encode temporal scope meaningfully

---

## 7. Dataset Specification

### Minimal Substitution Template Dataset

**Structure**: Each pair differs by exactly ONE phrase

**Template families**:

```python
templates = [
    # Agent + temporal marker + goal
    "What should {agent} prioritize for {temporal_marker} {goal}?",

    # Action + temporal marker + outcome
    "What actions will {verb} {temporal_marker} {outcome}?",

    # Planning + temporal marker + domain
    "How can we {verb} {domain} for {temporal_marker}?",

    # Decision + temporal marker + context
    "Should we {action} given {temporal_marker} {context}?",
]

temporal_markers_immediate = [
    "today's", "this week's", "current", "immediate",
    "right now", "this moment's", "present"
]

temporal_markers_longterm = [
    "future generations'", "our grandchildren's", "coming decades'",
    "long-term", "posterity's", "tomorrow's world"
]

# Generate 100 pairs using templates + markers
# Ensure exactly one difference per pair
```

### Quality Controls

**For each pair, verify**:
1. ☐ Identical sentence structure
2. ☐ Identical core vocabulary (>90% overlap)
3. ☐ Only temporal marker differs
4. ☐ No explicit time numbers ("5 years")
5. ☐ Natural and grammatical
6. ☐ Semantically coherent (not nonsensical)
7. ☐ No other confounds (length, complexity, sentiment)

---

## 8. Implementation Plan

### Week 1: Pre-Testing & Dataset Creation

**Day 1-2**: Adversarial testing
- Generate adversarial examples
- Test with existing probes
- Gate decision: proceed or redesign

**Day 3-4**: Controlled dataset
- Generate minimal-substitution pairs
- Manual quality control
- Final dataset: 100 pairs

**Day 5**: Validation
- Extract activations
- Verify data quality
- Train initial probes

### Week 2: Core Experiments

**Day 1-2**: Controlled probe training
- Proper train/test methodology
- Cross-validation
- Statistical analysis

**Day 3-4**: Performance validation
- Create decision tasks
- Extract model completions
- Correlate probe with behavior

**Day 5**: Analysis & interpretation

### Week 3: Circuit Analysis

**Day 1-2**: Attention analysis
- Pattern extraction
- Head importance ranking

**Day 3-4**: Causal testing
- Activation patching
- Head ablation

**Day 5**: Circuit documentation

### Week 4: Write-up

- Results synthesis
- Visualization
- Paper draft

---

## 9. Falsifiability

### Clear Falsification Criteria

**Hypothesis**: "GPT-2 develops semantic temporal scope representations"

**Would be falsified by**:
1. Adversarial test >80% (follows keywords blindly)
2. Implicit temporal test <55% (chance level)
3. Performance validation: no correlation (r < 0.3)
4. Circuit analysis: no specific heads found

**If falsified**: We've learned GPT-2 does keyword detection, not semantic temporal reasoning. **This is still valuable** - it sets bounds on model capabilities.

---

## 10. Expected Outcomes & Interpretation

### Outcome 1: Strong Semantic Understanding (Best Case)

- Controlled: 75% accuracy
- Adversarial: 55% (confused by conflicts)
- Implicit: 65% (generalizes)
- Circuit: 3-4 critical heads in layers 7-9

**Interpretation**: GPT-2 learns compositional temporal semantics. Specific circuit handles temporal scope integration. Publishable at top venue.

### Outcome 2: Weak Semantic + Strong Keyword (Likely)

- Controlled: 80% accuracy
- Adversarial: 70% (mostly follows keywords)
- Implicit: 58% (slight above chance)
- Circuit: Distributed, no clear heads

**Interpretation**: Primarily keyword detection with some contextual modulation. Honest null result. Publishable with caveats.

### Outcome 3: Pure Keyword Detection (Possible)

- Controlled: 95% accuracy
- Adversarial: 85% (follows keywords)
- Implicit: 50% (chance)
- Circuit: Early layer embeddings sufficient

**Interpretation**: No semantic temporal understanding. Model does surface pattern matching. Important negative result - shows limitations of probe-based interpretability.

---

## Conclusion

This rigorous design:
1. ✅ Controls confounds systematically
2. ✅ Uses performance validation (not just classification)
3. ✅ Includes adversarial testing upfront
4. ✅ Makes mechanistic predictions
5. ✅ Has clear success/failure criteria
6. ✅ Is falsifiable
7. ✅ Delivers value regardless of outcome

**Next step**: Run adversarial pre-test to gate decision on proceeding.

**Estimated timeline**: 3-4 weeks for complete rigorous study

**Expected contribution**: First principled study of temporal scope representation with proper confound controls and mechanistic validation.

---

**Ready to implement?** Let's start with adversarial dataset generation.
