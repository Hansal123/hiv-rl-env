"""
inference.py — OpenEnv RL Challenge Submission
HIV Drug Sequencing Environment Baseline Agent

This script evaluates a language model as a policy agent for HIV drug
sequencing. The agent receives patient observations and must select
optimal drug combinations to suppress viral load while minimizing
resistance development.

Output format strictly follows OpenEnv spec:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import sys
import json
import math
from openai import OpenAI
from environment import HIVDrugSequencingEnv, DRUG_COMBINATIONS, TaskGrader

# ─── Environment Variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


# ─── Agent Policy ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are an expert HIV clinician AI making antiretroviral therapy (ART) decisions.
Your goal is to select drug combinations that:
1. Suppress viral load below 400 copies/mL (ideally undetectable < 40)
2. Preserve or improve CD4 T-cell count
3. Minimize resistance mutation accumulation
4. Consider cross-resistance between drug classes

You must respond with ONLY a JSON object in this exact format:
{"action": <integer 0-311>, "reason": "<brief clinical rationale>"}

No other text. No markdown. Only the JSON object."""


def build_user_prompt(obs, step: int, task: str) -> str:
    # Get available drug combinations summary
    drug_info = []
    for i in range(0, 312, 30):  # Sample every 30th combo to keep prompt manageable
        c = DRUG_COMBINATIONS[i]
        drug_info.append(f"  [{i}] {c['name']} (class: NRTI+NRTI+{c['third_class']})")

    neighbour_text = ""
    if obs.neighbour_available and obs.neighbour_sequence:
        neighbour_text = f"""
NEAREST HISTORICAL PATIENT (similar CD4/VL/mutations):
  Their successful sequence: {' → '.join(obs.neighbour_sequence[:5])}
  Use this as a GUIDE but adapt to current resistance profile."""

    resistance_text = ", ".join(
        [cls for cls, resistant in obs.resistance_flags.items() if resistant]
    ) or "none"

    return f"""PATIENT STATUS (Step {step}, Task: {task.upper()}):
─────────────────────────────────────────
CD4 Count:        {obs.cd4_count:.0f} cells/mm³  {'⚠️ LOW' if obs.cd4_count < 200 else '✓ OK'}
Viral Load:       {obs.viral_load:.0f} copies/mL  {'✓ SUPPRESSED' if obs.viral_load < 400 else '⚠️ DETECTABLE'}
Mutations:        {obs.mutation_count}
Resistance to:    {resistance_text}
Days on regimen:  {obs.days_on_current_regimen}
Current class:    {obs.current_drug_class}
{neighbour_text}
AVAILABLE DRUG COMBINATIONS (sample — full range 0-311):
{chr(10).join(drug_info)}
... (312 total combinations available)

CLINICAL RULES:
- NEVER use a drug class marked as resistant
- If VL < 400 and stable, consider continuing current regimen
- If switching, prefer combinations with classes NOT yet resistant
- Hard task: every drug class matters — preserve options

Respond with JSON only: {{"action": <0-311>, "reason": "<rationale>"}}"""


def get_agent_action(obs, step: int, task: str, last_error: str = None) -> tuple[int, str]:
    """Query LLM for drug selection decision."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_prompt(obs, step, task)}
    ]

    if last_error:
        messages.append({
            "role": "user",
            "content": f"Previous action caused error: {last_error}. Please choose a valid action (0-311)."
        })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response
        parsed = json.loads(content)
        action = int(parsed.get("action", 0))
        reason = parsed.get("reason", "no reason given")
        action = max(0, min(311, action))
        return action, reason

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: extract first number from response
        import re
        numbers = re.findall(r'\b(\d{1,3})\b', content)
        for n in numbers:
            if 0 <= int(n) <= 311:
                return int(n), "parsed from malformed response"
        return 0, "fallback action"


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task: str, seed: int = 42) -> dict:
    """Run one full episode and return results."""
    env = HIVDrugSequencingEnv(task=task, seed=seed)
    obs = env.reset()

    benchmark = "hiv-drug-sequencing"
    print(f"[START] task={task} env={benchmark} model={MODEL_NAME}")

    rewards = []
    steps = 0
    done = False
    last_error = None
    success = False

    try:
        while not done:
            steps += 1

            # Get action from agent
            action, reason = get_agent_action(obs, steps, task, last_error)
            action_str = f"prescribe(combo={action},drug='{DRUG_COMBINATIONS[action]['name']}')"

            # Step environment
            try:
                obs, reward, done, info = env.step(action)
                last_error = None
                error_str = "null"
            except Exception as e:
                last_error = str(e)
                error_str = last_error
                reward = 0.0
                done = True

            rewards.append(reward)
            done_str = "true" if done else "false"
            print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

        # Grade final episode
        final_state = env.state()
        history = env._history

        grader = TaskGrader()
        if task == "easy":
            score = grader.grade_easy(history, final_state)
        elif task == "medium":
            score = grader.grade_medium(history, final_state)
        else:
            score = grader.grade_hard(history, final_state)

        success = score >= 0.5

    except Exception as e:
        error_str = str(e)
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={error_str}")
        score = 0.0
        success = False
    finally:
        env.close()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")

    return {
        "task": task,
        "success": success,
        "score": score,
        "steps": steps,
        "total_reward": sum(rewards),
        "rewards": rewards
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        result = run_episode(task=task, seed=42)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"Task: {r['task']:8s} | Score: {r['score']:.3f} | "
              f"Success: {str(r['success']):5s} | Steps: {r['steps']:2d} | "
              f"Total Reward: {r['total_reward']:.2f}")

    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.3f}")
    print("="*60)
