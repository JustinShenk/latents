# Research: Temporal Grounding Experiments

Scientific validation of temporal steering using linear probes and adversarial testing.

## Quick Start

### Run Experiments Locally

```bash
# Phase 0: Validate temporal encoding (78% probe accuracy)
python3 scripts/run_phase0.py

# Adversarial test: Validate steering (r=0.935 correlation)
source ../venv_steering/bin/activate
python3 tools/test_steering_with_probes.py
```

### Run on GCP (with GPU)

```bash
# 1. Upload code
./scripts/gcp_manager.sh upload

# 2. Create instance
./scripts/gcp_manager.sh create

# 3. SSH and setup
./scripts/gcp_manager.sh ssh
cd ~/temporal-grounding-gpt2/research
./scripts/setup_gcp.sh

# 4. Run Phase 0
./scripts/run_on_gcp.sh

# 5. Download results and delete instance
exit
./scripts/gcp_manager.sh download
./scripts/gcp_manager.sh delete  # IMPORTANT: Stop charges
```

## Results

- **Phase 0**: GPT-2 encodes temporal scope (Layer 8: 78% accuracy)
- **Adversarial Test**: Steering manipulates temporal features (Layer 11: r=0.935)
- See `docs/ADVERSARIAL_PROBE_TEST_RESULTS.md` for full analysis

## Structure

```
research/
├── tools/              # Analysis scripts
├── scripts/            # Experiment runners
├── docs/               # Research findings
├── data/               # Experimental data
├── results/            # Output files
└── probes/             # Trained classifiers
```
