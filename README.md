# 🧬 HIV Drug Sequencing RL Environment

> **OpenEnv RL Challenge Submission** — Policy-Mixture Reinforcement Learning for HIV Treatment Sequencing

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-brightgreen)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

---

## Overview & Motivation

HIV treatment is a sequential decision-making problem. As a patient undergoes therapy, the virus mutates and develops resistance to drugs — requiring periodic regimen switches. The key challenge:

> **Which drug sequence should be prescribed to maximise long-term viral suppression while preserving future drug options?**

This environment models that exact problem as an RL task. The agent observes the patient's clinical state (CD4, viral load, mutations) and selects from 312 antiretroviral drug combinations.

**The Policy-Mixture Innovation:**
- When a historically similar patient exists → the agent uses their drug sequence as guidance (**Neighbour Policy**)
- When no close match is found → the agent falls back to a model-based approach using resistance profiles (**Model-Based Policy**)

This mirrors real published research (Ernst 2005, Parbhoo 2014, Marivate 2015) on the EU Resist Database — and outperforms either strategy alone.

---

## Environment Specification

### Observation Space

| Field | Type | Description |
|---|---|---|
| `cd4_count` | float | CD4 T-cell count (cells/mm³), normal range 500–1500 |
| `viral_load` | float | HIV viral copies/mL (target: undetectable <40) |
| `mutation_count` | int | Resistance mutations accumulated |
| `treatment_step` | int | Current step in the treatment episode |
| `current_drug_class` | str | Drug class currently being used |
| `resistance_flags` | dict | Boolean resistance flags per drug class (NRTI/NNRTI/PI/INSTI/FI) |
| `neighbour_available` | bool | Whether a similar historical patient exists |
| `neighbour_sequence` | list\|null | Successful drug sequence from nearest neighbour patient |
| `days_on_current_regimen` | int | Days on current drug regimen |

### Action Space

- **Type:** Discrete
- **Size:** 312 combinations
- **Structure:** Each action is an index into 312 common antiretroviral combinations
- **Composition:** 2 NRTIs + 1 third agent (NNRTI, PI, INSTI, or FI)
- **Drugs:** 20 ARV drugs across 5 classes (ZDV, 3TC, TDF, FTC, ABC, d4T, ddI, EFV, NVP, ETR, RPV, LPV/r, ATV/r, DRV/r, SQV, RAL, EVG, DTG, BIC, ENF)

### Reward Function

Based on published HIV RL literature:

```
When Viral Load > 40 copies/mL:
  r_t = -0.7 × log(V_t) + 0.6 × log(T_t) - 0.2 × |M_t|

When Viral Load ≤ 40 (undetectable):
  r_t = 5 + 0.6 × log(T_t) - 0.2 × |M_t|

Additional:
  +2.0  if VL < 400 AND CD4 > 350  (treat-to-target)
  +4.0  if VL < 40                 (undetectable = optimal)
  -1.0  if unnecessary drug switch when already suppressed
```

**Anti-cheating properties:** Multi-component reward prevents gaming. Maximising viral suppression alone increases mutation penalty. Preserving drug classes is rewarded implicitly through sustained low mutations.

---

## Tasks

### Task 1: Easy — Treatment-Naive Patient
- **Baseline CD4:** 350–600 cells/mm³
- **Baseline VL:** 1,000–10,000 copies/mL  
- **Resistance:** 0–1 drug class
- **Max Steps:** 8 quarterly visits
- **Objective:** Achieve and maintain viral suppression (VL < 400)
- **Success threshold:** Score ≥ 0.5

**Grading:**
- 50% — proportion of steps with VL < 400
- 30% — final CD4 count (target > 400)
- 20% — final viral suppression bonus

### Task 2: Medium — Treatment-Experienced Patient  
- **Baseline CD4:** 200–400 cells/mm³
- **Baseline VL:** 10,000–100,000 copies/mL
- **Resistance:** 1–3 drug classes pre-existing
- **Max Steps:** 10 quarterly visits
- **Objective:** Suppress VL despite existing resistance without developing new resistance
- **Success threshold:** Score ≥ 0.5

**Grading:**
- 50% — final viral suppression score
- 25% — final CD4 count (target > 350)
- 25% — resistance mutations not worsened

### Task 3: Hard — Multi-Drug Resistant HIV
- **Baseline CD4:** 50–200 cells/mm³
- **Baseline VL:** 50,000–500,000 copies/mL
- **Resistance:** 3–7 classes (extensive pre-treatment)
- **Max Steps:** 12 quarterly visits
- **Objective:** Salvage therapy — any suppression + preserve drug classes + CD4 recovery
- **Success threshold:** Score ≥ 0.5

**Grading:**
- 50% — any viral suppression achieved (partial credit for VL < 10,000)
- 30% — drug classes preserved for future use
- 20% — CD4 count improved from baseline

---

## Setup & Usage

### Local Installation

```bash
git clone <repo>
cd hiv-rl-env
pip install -r requirements.txt
```

### Run Inference (all 3 tasks)

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1   # optional, has default
export MODEL_NAME=gpt-4.1-mini                   # optional, has default

python inference.py
```

### Run Web Interface

```bash
python app.py
# Opens at http://localhost:7860
```

### Docker

```bash
# Build
docker build -t hiv-rl-env .

# Run inference
docker run -e HF_TOKEN=your_token -e MODEL_NAME=gpt-4.1-mini hiv-rl-env python inference.py

# Run web interface
docker run -p 7860:7860 -e HF_TOKEN=your_token hiv-rl-env
```

### Use Environment Directly

```python
from environment import HIVDrugSequencingEnv

env = HIVDrugSequencingEnv(task="medium", seed=42)
obs = env.reset()

done = False
while not done:
    action = your_policy(obs)          # integer 0-311
    obs, reward, done, info = env.step(action)
    print(f"VL: {obs.viral_load:.0f}, CD4: {obs.cd4_count:.0f}, Reward: {reward:.2f}")

env.close()
```

---

## Baseline Performance Scores

Evaluated with GPT-4.1-mini using the heuristic + LLM policy-mixture agent:

| Task | Score | Success | Steps | Notes |
|------|-------|---------|-------|-------|
| Easy | 0.62 | ✅ | 8 | Suppression achieved by step 5 |
| Medium | 0.48 | ❌ | 10 | Struggles with cross-resistance |
| Hard | 0.31 | ❌ | 12 | Partial suppression only |
| **Average** | **0.47** | — | — | Room for significant improvement |

---

## Architecture

```
hiv-rl-env/
├── inference.py       # Main OpenEnv submission script
├── environment.py     # Core RL environment (Pydantic models, simulator, graders)
├── app.py             # Gradio web interface (HuggingFace Spaces)
├── openenv.yaml       # OpenEnv metadata
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Design Decisions

**Why HIV drug sequencing?**  
Real-world clinical problem with measurable outcomes, genuine sequential decision structure, and published RL benchmarks to compare against.

**Why 312 combinations?**  
Matches the EU Resist Database study (Ernst 2005) — 20 common ARV drugs forming standard 2-NRTI backbone + third agent regimens.

**Why Policy-Mixture?**  
From the lecture slides: Policy-Mixture scored 11.52 DR Reward vs 9.35 for Neighbour-only and 3.37 for Model-Based alone. The hybrid approach is empirically superior.

**Reward cheating prevention:**  
Multi-component reward, Conservative Q-Learning alignment, hard clinical constraints (never suggest fully resistant drug class), and mutation penalties that accumulate over time.

---

## References

- Ernst, D. et al. (2006). *Clinical data based optimal STI strategies for HIV*
- Parbhoo, S. et al. (2014). *Combining kernel and model based learning for HIV therapy selection*
- Marivate, V. et al. (2015). *Treatment of HIV/AIDS using reinforcement learning*
- EU Resist Database (EURESIST) — European HIV drug resistance database

---

## License

MIT License — see LICENSE for details.
