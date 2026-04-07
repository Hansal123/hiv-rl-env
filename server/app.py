"""
app.py — Hugging Face Spaces Web Interface
HIV Drug Sequencing RL Environment

Provides a visual dashboard to:
- Run inference episodes interactively
- Visualise patient state over time
- Compare policy strategies
- Inspect reward decomposition
"""

import os
import json
import math
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from environment import HIVDrugSequencingEnv, DRUG_COMBINATIONS, TaskGrader, HISTORICAL_PATIENTS

app_api = FastAPI()
global_env = None

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    action: int

@app_api.post("/reset")
def reset_endpoint(req: ResetRequest = None):
    global global_env
    if req is None: req = ResetRequest()
    global_env = HIVDrugSequencingEnv(task=req.task, seed=req.seed)
    obs = global_env.reset()
    return obs.dict() if hasattr(obs, 'dict') else obs.model_dump()

@app_api.post("/step")
def step_endpoint(req: StepRequest):
    global global_env
    if global_env is None:
        return {"error": "Environment not initialized"}
    obs, reward, done, info = global_env.step(int(req.action))
    return {
        "observation": obs.dict() if hasattr(obs, 'dict') else obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info.dict() if hasattr(info, 'dict') else info.model_dump()
    }

@app_api.get("/state")
def state_endpoint():
    global global_env
    if global_env is None:
         return {"error": "Environment not initialized"}
    return global_env.state()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def format_vl(vl: float) -> str:
    if vl < 40:   return f"<40 ✅ Undetectable"
    if vl < 400:  return f"{vl:.0f} ✅ Suppressed"
    if vl < 10000: return f"{vl:.0f} ⚠️ Detectable"
    return f"{vl:.0f} 🔴 High"

def format_cd4(cd4: float) -> str:
    if cd4 > 500: return f"{cd4:.0f} ✅ Normal"
    if cd4 > 350: return f"{cd4:.0f} 🟡 Low-normal"
    if cd4 > 200: return f"{cd4:.0f} ⚠️ Low"
    return f"{cd4:.0f} 🔴 Critical"

def run_demo_episode(task: str, seed: int) -> tuple:
    """Run a demo episode with a simple heuristic agent."""
    env = HIVDrugSequencingEnv(task=task, seed=int(seed))
    obs = env.reset()

    log_lines = []
    log_lines.append(f"🧬 HIV Drug Sequencing — Task: {task.upper()}")
    log_lines.append("=" * 55)
    log_lines.append(f"Patient ID: {env._state['patient_id']}")
    log_lines.append(f"Baseline CD4:        {format_cd4(obs.cd4_count)}")
    log_lines.append(f"Baseline Viral Load: {format_vl(obs.viral_load)}")
    log_lines.append(f"Baseline Mutations:  {obs.mutation_count}")
    log_lines.append(f"Neighbour found:     {'✅ Yes' if obs.neighbour_available else '❌ No (using model-based)'}")
    if obs.neighbour_available and obs.neighbour_sequence:
        log_lines.append(f"Neighbour sequence:  {' → '.join(obs.neighbour_sequence[:4])}")
    log_lines.append("")
    log_lines.append("TREATMENT LOG:")
    log_lines.append("-" * 55)

    done = False
    step = 0
    total_reward = 0
    history_rows = []

    while not done:
        step += 1

        # Heuristic policy: use neighbour sequence if available, else pick best class
        if obs.neighbour_available and obs.neighbour_sequence:
            seq_idx = min(step - 1, len(obs.neighbour_sequence) - 1)
            neighbour_drug = obs.neighbour_sequence[seq_idx]
            # Find combo index matching neighbour suggestion
            action = next(
                (i for i, c in enumerate(DRUG_COMBINATIONS) if c["name"] == neighbour_drug),
                step * 13 % 312
            )
        else:
            # Model-based: rotate through drug classes avoiding resistant ones
            resistant_classes = [cls for cls, r in obs.resistance_flags.items() if r]
            safe_combos = [
                i for i, c in enumerate(DRUG_COMBINATIONS)
                if c["third_class"] not in resistant_classes
            ]
            action = safe_combos[step % len(safe_combos)] if safe_combos else step % 312

        obs, reward, done, info = env.step(action)
        total_reward += reward

        combo_name = DRUG_COMBINATIONS[action]["name"]
        log_lines.append(
            f"Step {step:2d} | {combo_name:28s} | "
            f"VL: {obs.viral_load:8.0f} | CD4: {obs.cd4_count:5.0f} | "
            f"Reward: {reward:+.2f}"
        )

        history_rows.append([
            step,
            combo_name,
            f"{obs.viral_load:.0f}",
            f"{obs.cd4_count:.0f}",
            obs.mutation_count,
            f"{reward:.2f}",
            "✅" if obs.viral_load < 400 else "❌"
        ])

    # Grade
    final_state = env.state()
    history = env._history
    grader = TaskGrader()
    if task == "easy":
        score = grader.grade_easy(history, final_state)
    elif task == "medium":
        score = grader.grade_medium(history, final_state)
    else:
        score = grader.grade_hard(history, final_state)

    log_lines.append("-" * 55)
    log_lines.append(f"FINAL OUTCOME:")
    log_lines.append(f"  Viral Load:    {format_vl(final_state['viral_load'])}")
    log_lines.append(f"  CD4 Count:     {format_cd4(final_state['cd4'])}")
    log_lines.append(f"  Mutations:     {final_state['mutations']}")
    log_lines.append(f"  Total Reward:  {total_reward:.2f}")
    log_lines.append(f"  Episode Score: {score:.3f} {'✅ SUCCESS' if score >= 0.5 else '❌ FAILED'}")
    log_lines.append(f"  Steps taken:   {step}")

    env.close()

    headers = ["Step", "Drug Combination", "Viral Load", "CD4", "Mutations", "Reward", "Suppressed"]
    return "\n".join(log_lines), gr.DataFrame(value=history_rows, headers=headers)


def get_env_info() -> str:
    info = f"""
## HIV Drug Sequencing RL Environment

**What this environment models:**
The clinical challenge of selecting optimal antiretroviral therapy (ART) 
sequences for HIV+ patients over multiple treatment visits.

**State Space:**
- CD4 T-cell count (immune health indicator)
- Viral Load (HIV copies/mL — lower is better)
- Resistance mutations accumulated
- Current drug class in use
- Nearest historical patient (neighbour guidance)

**Action Space:**
- 312 drug combinations (20 ARV drugs)
- Each combination: 2 NRTIs + 1 third agent (NNRTI/PI/INSTI/FI)

**Reward Function (from HIV RL literature):**
```
If VL > 40:  r = -0.7·log(VL) + 0.6·log(CD4) - 0.2·|mutations|
If VL ≤ 40:  r = 5 + 0.6·log(CD4) - 0.2·|mutations|
```
Plus bonuses for sustained suppression, penalties for unnecessary switches.

**Tasks:**
| Task   | CD4 Range  | Viral Load    | Resistance | Steps |
|--------|-----------|---------------|------------|-------|
| Easy   | 350–600   | 1k–10k        | 0–1 class  | 8     |
| Medium | 200–400   | 10k–100k      | 1–3 class  | 10    |
| Hard   | 50–200    | 50k–500k      | 3–7 class  | 12    |

**Policy-Mixture Strategy:**
- When similar historical patients exist → **Neighbour Policy** guides drug selection
- When no close neighbours found → **Model-Based Policy** relies on resistance profile

**Historical Database:** {len(HISTORICAL_PATIENTS)} synthetic patient records for neighbour lookup
"""
    return info


# ─── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks(
    title="HIV Drug Sequencing RL Environment"
) as demo:

    gr.HTML("""
    <div class="header-box">
        <h1 style="margin:0; font-size:28px; font-weight:700; color:white;">
            🧬 HIV Drug Sequencing RL Environment
        </h1>
        <p style="margin:8px 0 0 0; opacity:0.85; font-size:15px; color:white;">
            OpenEnv Challenge — Policy-Mixture Reinforcement Learning for Clinical Decision Support
        </p>
        <p style="margin:4px 0 0 0; opacity:0.7; font-size:13px; color:white;">
            Agent: Selects optimal ART sequences to suppress viral load while minimizing drug resistance
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Run Episode ─────────────────────────────────────────────
        with gr.Tab("▶ Run Episode"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dropdown = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="easy",
                        label="Task Difficulty",
                        info="Easy: treatment-naive | Medium: experienced | Hard: multi-drug resistant"
                    )
                    seed_slider = gr.Slider(
                        minimum=1, maximum=999, value=42, step=1,
                        label="Random Seed",
                        info="Change seed to get different patient profiles"
                    )
                    run_btn = gr.Button("🚀 Run Episode", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML("""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px;">
                        <h3 style="margin:0 0 8px 0; color:#1e293b;">How the Agent Works</h3>
                        <p style="margin:0; color:#475569; font-size:14px; line-height:1.6;">
                        <b>1. Neighbour Check:</b> Searches 500 historical patients for similar CD4/VL/mutations<br>
                        <b>2. If neighbour found:</b> Uses their successful drug sequence as guidance<br>
                        <b>3. If no neighbour:</b> Falls back to model-based policy avoiding resistant classes<br>
                        <b>4. Reward signal:</b> Composite formula rewarding suppression + CD4 - mutations
                        </p>
                    </div>
                    """)

            episode_log = gr.Textbox(
                label="Episode Log",
                lines=20,
                max_lines=30
            )
            episode_table = gr.DataFrame(
                label="Step-by-Step Treatment History",
                headers=["Step", "Drug Combination", "Viral Load", "CD4", "Mutations", "Reward", "Suppressed"]
            )

            run_btn.click(
                fn=run_demo_episode,
                inputs=[task_dropdown, seed_slider],
                outputs=[episode_log, episode_table]
            )

        # ── Tab 2: Environment Info ────────────────────────────────────────
        with gr.Tab("📋 Environment Info"):
            gr.Markdown(get_env_info())

        # ── Tab 3: Drug Database ───────────────────────────────────────────
        with gr.Tab("💊 Drug Combinations"):
            gr.Markdown("### Available Drug Combinations (sample of 312 total)")
            sample_combos = [
                [i, c["name"], c["nrti1"], c["nrti2"], c["third"], c["third_class"]]
                for i, c in enumerate(DRUG_COMBINATIONS[:50])
            ]
            gr.DataFrame(
                value=sample_combos,
                headers=["Index", "Combination", "NRTI 1", "NRTI 2", "Third Agent", "Third Class"],
                label="First 50 of 312 combinations"
            )

        # ── Tab 4: Reward Function ─────────────────────────────────────────
        with gr.Tab("📊 Reward Function"):
            gr.Markdown("""
### Reward Function (based on HIV RL literature)

The reward at each step is calculated as:

**When Viral Load > 40 copies/mL:**
```
r_t = -0.7 × log(V_t) + 0.6 × log(T_t) - 0.2 × |M_t|
```

**When Viral Load ≤ 40 (undetectable):**
```
r_t = 5 + 0.6 × log(T_t) - 0.2 × |M_t|
```

Where:
- **V_t** = Viral Load at time t
- **T_t** = CD4 count at time t  
- **M_t** = Number of resistance mutations

**Additional components:**
- **+2.0** bonus when VL < 400 AND CD4 > 350 ("treat to target" achieved)
- **+4.0** bonus when VL < 40 (undetectable = optimal outcome)
- **-1.0** penalty for unnecessary drug switch when already suppressed
- **Resistance development** automatically increases |M_t| penalty

**Why this prevents reward cheating:**
The multi-component formula means the agent cannot game a single metric.
Maximising drug dose might suppress viral load but increases mutations.
Keeping resistance low preserves future drug options.
""")

    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#94a3b8; font-size:13px; border-top:1px solid #e2e8f0; margin-top:16px;">
        HIV Drug Sequencing RL Environment — OpenEnv Challenge Submission<br>
        Based on: Ernst 2005 · Parbhoo 2014 · Marivate 2015 · EU Resist Database methodology
    </div>
    """)

app_api = gr.mount_gradio_app(app_api, demo, path="/")

def main_server():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app_api, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main_server()
