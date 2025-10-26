# Package Naming Strategy for ML Researchers

## Problem Statement

Need to rename from `temporal-steering` to attract ML researchers while:
1. Reflecting the **multi-dimensional** steering capability (not just temporal)
2. Being **short** and **memorable**
3. **Available** on PyPI (`horizon` is taken)
4. Appealing to **scientific/research** audience

## Naming Criteria

### What ML Researchers Value:
- ✅ **Clarity**: Immediately understand what the package does
- ✅ **Scientific**: References methods/papers (e.g., "CAA")
- ✅ **Professional**: Not too cute, not too generic
- ✅ **Short imports**: `from X import ...` should be concise
- ✅ **Searchable**: Unique enough to find in Google/arXiv

### Anti-patterns:
- ❌ Too generic (`ml-tools`, `ai-utils`)
- ❌ Too cute (`steer-buddy`, `llm-pilot`)
- ❌ Too long (`temporal-steering-framework-for-llms`)
- ❌ Unclear purpose (`project-x`, `framework2`)

## Top Recommendations

### Option 1: `steering-vectors` ⭐⭐⭐⭐⭐

**Import**: `from steering_vectors import SteeringFramework`

**Pros**:
- ✅ Describes exactly what it provides (steering vectors)
- ✅ Professional, scientific tone
- ✅ Searchable (unique phrase in ML)
- ✅ Aligns with literature (e.g., "activation steering", "representation engineering")
- ✅ Works for temporal + all other dimensions

**Cons**:
- ⚠️ May be taken on PyPI (need to check)
- ⚠️ Slightly generic (but specific enough)

**Tagline**: *"Steer LLMs along multiple behavioral dimensions using activation vectors"*

**Target audience**: Researchers working on:
- Activation steering
- Representation engineering
- Mechanistic interpretability
- LLM control/safety

### Option 2: `repere` (French: "reference point") ⭐⭐⭐⭐

**Import**: `from repere import SteeringFramework`

**Pros**:
- ✅ Very short (6 letters)
- ✅ Unique, memorable
- ✅ Elegant/sophisticated
- ✅ Metaphor: Reference points for navigating activation space
- ✅ Almost certainly available on PyPI

**Cons**:
- ⚠️ Not immediately obvious what it does
- ⚠️ Requires explanation ("repere = steering LLMs")
- ⚠️ May confuse non-French speakers

**Tagline**: *"Navigate activation space with multi-dimensional steering vectors"*

**Target audience**: Researchers who value:
- Elegant abstractions
- Interpretability research
- Novel framing of steering

### Option 3: `caa-steering` ⭐⭐⭐⭐

**Import**: `from caa_steering import SteeringFramework`

**Pros**:
- ✅ References Contrastive Activation Addition (CAA) method
- ✅ Immediately signals to researchers familiar with CAA literature
- ✅ Professional, academic tone
- ✅ Likely available

**Cons**:
- ⚠️ Limits scope to CAA method (what if we add other methods?)
- ⚠️ Excludes researchers not familiar with CAA

**Tagline**: *"Multi-dimensional CAA steering for LLMs"*

**Target audience**: Researchers specifically working on:
- CAA / activation steering
- AI alignment
- LLM safety

### Option 4: `steervec` ⭐⭐⭐

**Import**: `from steervec import SteeringFramework`

**Pros**:
- ✅ Very short (8 letters)
- ✅ Clear meaning (steering vectors)
- ✅ Easy to remember
- ✅ Informal but professional

**Cons**:
- ⚠️ Slightly informal (contraction)
- ⚠️ May feel less academic

**Tagline**: *"Steering vectors for multi-dimensional LLM control"*

### Option 5: `activsteer` ⭐⭐⭐

**Import**: `from activsteer import SteeringFramework`

**Pros**:
- ✅ References "activation steering" (known technique)
- ✅ Short
- ✅ Professional

**Cons**:
- ⚠️ "Activ" might feel like incomplete word
- ⚠️ Less elegant than alternatives

## Availability Check (Simulate)

```bash
# Check PyPI availability
pip index versions <package-name>

# Check if taken:
# - steering-vectors: ?
# - repere: ?
# - caa-steering: ?
# - steervec: ?
# - activsteer: ?
```

## Recommended Choice: `steering-vectors`

### Why This Wins:

1. **Clarity**: Researchers immediately know it's about steering via vectors
2. **Scope**: Works for temporal + all future dimensions
3. **Alignment**: Matches existing terminology ("activation vectors", "steering vectors")
4. **Professionalism**: Not too technical, not too casual
5. **Discovery**: Easy to find in papers/GitHub/blogs

### Branding Strategy:

**Package name**: `steering-vectors` (PyPI)
**Import name**: `steering_vectors` (Python)
**GitHub repo**: `steering-vectors`
**Domain** (if needed): `steeringvectors.ai` or `.org`

**Primary features**:
- **Temporal Scope**: Flagship dimension (immediate ↔ long-term)
- **Plugin System**: Community can add dimensions
- **Multi-Model**: GPT-2, LLaMA, Mistral, etc.

**Tagline options**:
1. *"Steer LLMs with multi-dimensional activation vectors"*
2. *"Control LLM behavior along any dimension using CAA"*
3. *"The extensible framework for LLM steering research"*

**Positioning**:
- **vs nrimsky/CAA**: More general (not just safety behaviors), multi-model
- **vs steering-vectors (if exists)**: Focuses on research/production-ready tools
- **Unique value**: Plugin architecture + temporal steering

### Marketing to ML Researchers:

**Paper abstract** (hypothetical):
> "We present steering-vectors, an extensible framework for controlling large language model behavior along multiple dimensions using Contrastive Activation Addition (CAA). Our flagship dimension, temporal scope, enables steering from immediate to long-term thinking..."

**GitHub README** (first lines):
```markdown
# Steering Vectors

Control LLM outputs along multiple behavioral dimensions using activation steering.

- 🎯 **Temporal Scope**: Immediate ↔ Long-term thinking (flagship feature)
- 🔌 **Extensible**: Plugin system for custom dimensions
- 🚀 **Multi-Model**: GPT-2, LLaMA, Mistral, Falcon, and more
- 🔬 **Research-Ready**: PCA analysis, confound testing, human evaluation tools
```

**First code example**:
```python
from steering_vectors import SteeringFramework

framework = SteeringFramework.load(model, tokenizer, "temporal_scope")

# Generate with long-term thinking
result = framework.generate(
    prompt="How should we address climate change?",
    steerings=[("temporal_scope", 0.8)],
    temperature=0.7
)
```

## Alternative: If `steering-vectors` is Taken

### Fallback: `repere`

**Branding**:
- **Name**: Repère (French: reference point, landmark)
- **Metaphor**: Navigation through activation space
- **Pronunciation**: "ruh-PAIR" (help researchers pronounce it)

**Marketing**:
```markdown
# Repère

Navigate LLM behavior with multi-dimensional steering vectors.

*Repère* (French: reference point) provides a framework for steering large
language models along behavioral dimensions like temporal scope, formality,
optimism, and more.
```

## Timeline

1. **Now**: Check PyPI availability
2. **Day 1**: Choose name, update package files
3. **Day 2**: Update all docs/examples
4. **Day 3**: Publish to PyPI
5. **Week 1**: Announce on Twitter/LinkedIn/arXiv

## Questions for Team

1. Do we want to emphasize CAA method or be method-agnostic?
2. Priority: Immediate recognition or long-term brand building?
3. Target audience: Pure researchers or also practitioners?
4. Will we publish a paper? (influences naming)

---

**Recommendation**: Go with `steering-vectors` if available, fallback to `repere` if not.

**Why**: Attracts ML researchers by being clear, professional, and aligned with existing terminology, while still being unique enough to own the space.
