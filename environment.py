"""
HIV Drug Sequencing RL Environment
===================================
Models HIV treatment sequencing as a reinforcement learning problem.
The agent must learn optimal drug combination sequences to suppress
viral load while minimizing resistance development.

State:  CD4 count, Viral Load, Mutation profile, Treatment history
Action: Drug combination index (from 312 common combinations of 20 drugs)
Reward: Composite clinical reward (viral suppression + CD4 health - mutations)
"""

import random
import math
import json
import copy
from typing import Optional
from pydantic import BaseModel, Field


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class PatientObservation(BaseModel):
    """Current patient state observable by the agent."""
    cd4_count: float = Field(..., description="CD4 T-cell count (cells/mm³), normal 500-1500")
    viral_load: float = Field(..., description="HIV viral copies/mL (undetectable <40)")
    mutation_count: int = Field(..., description="Number of resistance mutations accumulated")
    treatment_step: int = Field(..., description="Current step in treatment episode")
    current_drug_class: str = Field(..., description="Current drug class in use")
    resistance_flags: dict = Field(..., description="Per-drug-class resistance flags")
    neighbour_available: bool = Field(..., description="Whether a similar historical patient exists")
    neighbour_sequence: Optional[list] = Field(None, description="Drug sequence from nearest neighbour")
    days_on_current_regimen: int = Field(..., description="Days patient has been on current regimen")


class DrugAction(BaseModel):
    """Drug combination action taken by the agent."""
    drug_combination_index: int = Field(..., description="Index into 312 drug combinations (0-311)")
    drug_name: str = Field(..., description="Human-readable drug combination name")
    switch_reason: Optional[str] = Field(None, description="Reason for switching if applicable")


class ClinicalReward(BaseModel):
    """Decomposed reward signal for interpretability."""
    total: float
    viral_suppression_component: float
    cd4_health_component: float
    resistance_penalty: float
    switch_penalty: float
    clinical_bonus: float


class EpisodeInfo(BaseModel):
    """Additional episode information."""
    step: int
    patient_id: str
    reward_breakdown: ClinicalReward
    treatment_failed: bool
    resistance_developed: str
    virological_suppression: bool


# ─── Drug Definitions ─────────────────────────────────────────────────────────

DRUG_CLASSES = {
    "NRTI": ["ZDV", "3TC", "TDF", "FTC", "ABC", "d4T", "ddI"],   # 7 drugs
    "NNRTI": ["EFV", "NVP", "ETR", "RPV"],                         # 4 drugs
    "PI": ["LPV/r", "ATV/r", "DRV/r", "SQV"],                     # 4 drugs
    "INSTI": ["RAL", "EVG", "DTG", "BIC"],                         # 4 drugs
    "FI": ["ENF"]                                                   # 1 drug = 20 total
}

ALL_DRUGS = []
for cls, drugs in DRUG_CLASSES.items():
    for d in drugs:
        ALL_DRUGS.append((d, cls))

# Generate 312 combinations: each combination = 2 NRTIs + 1 from another class
DRUG_COMBINATIONS = []
nrtis = [d for d, c in ALL_DRUGS if c == "NRTI"]
third_agents = [d for d, c in ALL_DRUGS if c != "NRTI"]

for i, n1 in enumerate(nrtis):
    for n2 in nrtis[i+1:]:
        for third in third_agents:
            DRUG_COMBINATIONS.append({
                "name": f"{n1}+{n2}+{third}",
                "nrti1": n1,
                "nrti2": n2,
                "third": third,
                "third_class": next(c for d, c in ALL_DRUGS if d == third)
            })

# Pad/trim to exactly 312
DRUG_COMBINATIONS = DRUG_COMBINATIONS[:312]
while len(DRUG_COMBINATIONS) < 312:
    DRUG_COMBINATIONS.append(DRUG_COMBINATIONS[len(DRUG_COMBINATIONS) % len(DRUG_COMBINATIONS)])


# ─── Historical Patient Database (Synthetic) ──────────────────────────────────

def generate_historical_patients(n: int = 500, seed: int = 42) -> list:
    """Generate synthetic historical patient records for neighbour lookup."""
    rng = random.Random(seed)
    patients = []
    for i in range(n):
        baseline_cd4 = rng.uniform(150, 600)
        baseline_vl = rng.uniform(1000, 100000)
        n_mutations = rng.randint(0, 5)

        # Generate a treatment sequence (4-8 steps)
        sequence = []
        for step in range(rng.randint(4, 8)):
            combo_idx = rng.randint(0, len(DRUG_COMBINATIONS) - 1)
            outcome_cd4 = baseline_cd4 + rng.uniform(-50, 150) * (step + 1) * 0.3
            outcome_vl = max(10, baseline_vl * math.exp(-rng.uniform(0.1, 0.8) * (step + 1)))
            sequence.append({
                "step": step,
                "combo_idx": combo_idx,
                "combo_name": DRUG_COMBINATIONS[combo_idx]["name"],
                "cd4": outcome_cd4,
                "viral_load": outcome_vl,
                "mutations": n_mutations + rng.randint(0, 2)
            })

        patients.append({
            "patient_id": f"HIST_{i:04d}",
            "baseline_cd4": baseline_cd4,
            "baseline_vl": baseline_vl,
            "baseline_mutations": n_mutations,
            "sequence": sequence,
            "final_suppressed": outcome_vl < 400
        })
    return patients


HISTORICAL_PATIENTS = generate_historical_patients(500)


# ─── Core Environment ─────────────────────────────────────────────────────────

class HIVDrugSequencingEnv:
    """
    HIV Drug Sequencing Environment.

    Models the clinical challenge of selecting optimal antiretroviral
    therapy (ART) sequences for HIV+ patients, accounting for:
    - Viral load suppression
    - CD4 T-cell preservation
    - Resistance mutation accumulation
    - Cross-resistance between drug classes
    - Neighbour-based treatment guidance (when available)
    """

    MAX_STEPS = 12          # ~12 quarterly visits = 3 years of treatment
    SUPPRESSION_THRESHOLD = 400  # viral copies/mL

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self._state = None
        self._step_count = 0
        self._done = False
        self._history = []

        # Task configurations
        self.task_configs = {
            "easy": {
                "baseline_cd4_range": (350, 600),
                "baseline_vl_range": (1000, 10000),
                "baseline_mutations": (0, 1),
                "resistance_rate": 0.1,
                "max_steps": 8,
                "description": "Treatment-naive patient with moderate viral load"
            },
            "medium": {
                "baseline_cd4_range": (200, 400),
                "baseline_vl_range": (10000, 100000),
                "baseline_mutations": (1, 3),
                "resistance_rate": 0.2,
                "max_steps": 10,
                "description": "Treatment-experienced patient with moderate resistance"
            },
            "hard": {
                "baseline_cd4_range": (50, 200),
                "baseline_vl_range": (50000, 500000),
                "baseline_mutations": (3, 7),
                "resistance_rate": 0.35,
                "max_steps": 12,
                "description": "Multi-drug resistant HIV with advanced immunosuppression"
            }
        }

        cfg = self.task_configs.get(task, self.task_configs["easy"])
        self.MAX_STEPS = cfg["max_steps"]
        self._cfg = cfg

    def reset(self) -> PatientObservation:
        """Reset environment to a new patient episode."""
        cfg = self._cfg
        self.rng = random.Random(self.seed)

        self._state = {
            "patient_id": f"PT_{self.rng.randint(10000, 99999)}",
            "cd4": self.rng.uniform(*cfg["baseline_cd4_range"]),
            "viral_load": self.rng.uniform(*cfg["baseline_vl_range"]),
            "mutations": self.rng.randint(*cfg["baseline_mutations"]),
            "resistance": {cls: False for cls in DRUG_CLASSES},
            "current_drug_class": "None",
            "days_on_regimen": 0,
            "treatment_history": [],
            "step": 0
        }

        self._step_count = 0
        self._done = False
        self._history = []

        # Find neighbour
        neighbour = self._find_neighbour()

        return PatientObservation(
            cd4_count=self._state["cd4"],
            viral_load=self._state["viral_load"],
            mutation_count=self._state["mutations"],
            treatment_step=0,
            current_drug_class="None",
            resistance_flags=self._state["resistance"],
            neighbour_available=neighbour is not None,
            neighbour_sequence=[s["combo_name"] for s in neighbour["sequence"]] if neighbour else None,
            days_on_current_regimen=0
        )

    def step(self, action: int) -> tuple:
        """
        Execute one treatment decision.

        Args:
            action: Drug combination index (0-311)

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        action = max(0, min(311, int(action)))
        combo = DRUG_COMBINATIONS[action]

        self._step_count += 1
        prev_state = copy.deepcopy(self._state)

        # ── Simulate clinical response ──────────────────────────────────────
        self._simulate_treatment_response(combo)

        # ── Calculate reward ────────────────────────────────────────────────
        reward_obj = self._calculate_reward(prev_state, combo)
        reward = reward_obj.total

        # ── Check termination ───────────────────────────────────────────────
        done = self._check_done()
        self._done = done

        # ── Build observation ───────────────────────────────────────────────
        neighbour = self._find_neighbour()
        obs = PatientObservation(
            cd4_count=self._state["cd4"],
            viral_load=self._state["viral_load"],
            mutation_count=self._state["mutations"],
            treatment_step=self._step_count,
            current_drug_class=combo["third_class"],
            resistance_flags=self._state["resistance"],
            neighbour_available=neighbour is not None,
            neighbour_sequence=[s["combo_name"] for s in neighbour["sequence"]] if neighbour else None,
            days_on_current_regimen=self._state["days_on_regimen"]
        )

        info = EpisodeInfo(
            step=self._step_count,
            patient_id=self._state["patient_id"],
            reward_breakdown=reward_obj,
            treatment_failed=self._state["viral_load"] > 100000 and self._step_count > 3,
            resistance_developed=self._get_resistance_summary(),
            virological_suppression=self._state["viral_load"] < self.SUPPRESSION_THRESHOLD
        )

        self._history.append({
            "step": self._step_count,
            "action": action,
            "combo": combo["name"],
            "cd4": self._state["cd4"],
            "viral_load": self._state["viral_load"],
            "mutations": self._state["mutations"],
            "reward": reward
        })

        return obs, reward, done, info

    def state(self) -> dict:
        """Return full internal state."""
        return copy.deepcopy(self._state)

    def close(self):
        """Clean up resources."""
        pass

    # ── Private Methods ────────────────────────────────────────────────────────

    def _simulate_treatment_response(self, combo: dict):
        """Simulate patient response to drug combination."""
        s = self._state
        cfg = self._cfg

        third_class = combo["third_class"]
        is_resistant = s["resistance"].get(third_class, False)
        nrti_resistant = s["resistance"].get("NRTI", False)

        # Base viral load response
        if is_resistant or nrti_resistant:
            # Resistant: poor suppression, possible rebound
            vl_factor = self.rng.uniform(0.8, 1.3)
        else:
            # Sensitive: good suppression
            vl_factor = self.rng.uniform(0.1, 0.6)

        s["viral_load"] = max(20, s["viral_load"] * vl_factor)

        # CD4 response (inversely related to viral load)
        if s["viral_load"] < 400:
            cd4_delta = self.rng.uniform(10, 80)
        elif s["viral_load"] < 10000:
            cd4_delta = self.rng.uniform(-20, 30)
        else:
            cd4_delta = self.rng.uniform(-60, -10)

        s["cd4"] = max(10, s["cd4"] + cd4_delta)

        # Resistance development (probabilistic)
        resistance_roll = self.rng.random()
        if resistance_roll < cfg["resistance_rate"] and s["viral_load"] > 1000:
            # Develop resistance to current third agent class
            s["resistance"][third_class] = True
            s["mutations"] += self.rng.randint(1, 2)

            # Cross-resistance: some classes share resistance pathways
            if third_class == "NNRTI" and resistance_roll < cfg["resistance_rate"] * 0.3:
                s["resistance"]["NRTI"] = True

        # Update days on regimen
        s["days_on_regimen"] += 90  # quarterly visits
        s["current_drug_class"] = third_class
        s["step"] = self._step_count

        # Track history
        s["treatment_history"].append(combo["name"])

    def _calculate_reward(self, prev_state: dict, combo: dict) -> ClinicalReward:
        """
        Composite clinical reward function.

        Based on: r_t = -0.7*log(V_t) + 0.6*log(T_t) - 0.2*|M_t|
        (from the HIV RL literature, Image 4 in lecture slides)
        """
        s = self._state
        vl = max(1, s["viral_load"])
        cd4 = max(1, s["cd4"])
        mutations = s["mutations"]

        # Core reward formula (from literature)
        if vl > 40:
            viral_component = -0.7 * math.log(vl)
            cd4_component = 0.6 * math.log(cd4)
            resistance_penalty = -0.2 * mutations
        else:
            # Virologically suppressed — bonus formula
            viral_component = 5.0
            cd4_component = 0.6 * math.log(cd4)
            resistance_penalty = -0.2 * mutations

        # Penalize unnecessary drug switches
        switch_penalty = 0.0
        if len(s["treatment_history"]) > 1:
            prev_combo = s["treatment_history"][-2] if len(s["treatment_history"]) >= 2 else None
            if prev_combo and prev_combo == combo["name"]:
                switch_penalty = 0.0  # staying on same regimen when working = no penalty
            elif prev_state["viral_load"] < 400:
                switch_penalty = -1.0  # switching when suppressed = penalize

        # Bonus for sustained suppression
        clinical_bonus = 0.0
        if vl < 400 and cd4 > 350:
            clinical_bonus = 2.0  # "Treat to target" achieved
        if vl < 40:
            clinical_bonus = 4.0  # Undetectable = optimal

        total = viral_component + cd4_component + resistance_penalty + switch_penalty + clinical_bonus

        return ClinicalReward(
            total=round(total, 4),
            viral_suppression_component=round(viral_component, 4),
            cd4_health_component=round(cd4_component, 4),
            resistance_penalty=round(resistance_penalty, 4),
            switch_penalty=round(switch_penalty, 4),
            clinical_bonus=round(clinical_bonus, 4)
        )

    def _check_done(self) -> bool:
        """Check episode termination conditions."""
        s = self._state

        # Max steps reached
        if self._step_count >= self.MAX_STEPS:
            return True

        # Patient death / treatment failure (CD4 < 10 is critically low)
        if s["cd4"] < 10:
            return True

        # All drug classes resistant (no options left)
        all_resistant = all(s["resistance"].get(cls, False) for cls in ["NRTI", "NNRTI", "PI", "INSTI"])
        if all_resistant:
            return True

        return False

    def _find_neighbour(self) -> Optional[dict]:
        """
        Find nearest historical patient using Euclidean distance
        in (CD4, log(VL), mutations) space.
        """
        if not self._state:
            return None

        s = self._state
        best_dist = float("inf")
        best_patient = None

        # Normalize features
        for p in HISTORICAL_PATIENTS:
            dist = math.sqrt(
                ((s["cd4"] - p["baseline_cd4"]) / 500) ** 2 +
                ((math.log(max(1, s["viral_load"])) - math.log(max(1, p["baseline_vl"]))) / 10) ** 2 +
                ((s["mutations"] - p["baseline_mutations"]) / 10) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_patient = p

        # Only return neighbour if close enough (threshold = 0.5 in normalized space)
        if best_dist < 0.5:
            return best_patient
        return None

    def _get_resistance_summary(self) -> str:
        """Get human-readable resistance summary."""
        resistant = [cls for cls, val in self._state["resistance"].items() if val]
        return ", ".join(resistant) if resistant else "none"


# ─── Task Graders ─────────────────────────────────────────────────────────────

class TaskGrader:
    """Programmatic grader for each task difficulty."""

    @staticmethod
    def grade_easy(history: list, final_state: dict) -> float:
        """
        Easy task grading:
        - Primary: achieve viral suppression (VL < 400) by step 6
        - Secondary: maintain CD4 > 300
        Score: 0.0 to 1.0
        """
        if not history:
            return 0.01

        suppressed_steps = sum(1 for h in history if h["viral_load"] < 400)
        final_vl = final_state.get("viral_load", 999999)
        final_cd4 = final_state.get("cd4", 0)

        suppression_score = min(1.0, suppressed_steps / max(1, len(history)) * 1.5)
        cd4_score = min(1.0, final_cd4 / 400)
        vl_bonus = 0.2 if final_vl < 400 else 0.0

        score = suppression_score * 0.5 + cd4_score * 0.3 + vl_bonus
        return max(0.01, min(0.99, score))

    @staticmethod
    def grade_medium(history: list, final_state: dict) -> float:
        """
        Medium task grading:
        - Primary: suppress VL despite existing resistance
        - Secondary: avoid developing new resistance mutations
        - Tertiary: preserve CD4
        """
        if not history:
            return 0.01

        final_vl = final_state.get("viral_load", 999999)
        final_cd4 = final_state.get("cd4", 0)
        final_mutations = final_state.get("mutations", 99)
        initial_mutations = history[0].get("mutations", 0) if history else 0

        suppression_score = 1.0 if final_vl < 400 else max(0, 1 - math.log10(max(1, final_vl)) / 6)
        cd4_score = min(1.0, final_cd4 / 350)
        mutation_score = max(0, 1 - (final_mutations - initial_mutations) / 5)

        score = suppression_score * 0.5 + cd4_score * 0.25 + mutation_score * 0.25
        return max(0.01, min(0.99, score))

    @staticmethod
    def grade_hard(history: list, final_state: dict) -> float:
        """
        Hard task grading:
        - Salvage therapy for multi-drug resistant HIV
        - Any suppression is meaningful
        - Must preserve at least one drug class for future use
        - CD4 recovery is the gold standard
        """
        if not history:
            return 0.01

        final_vl = final_state.get("viral_load", 999999)
        final_cd4 = final_state.get("cd4", 0)
        resistance = final_state.get("resistance", {})

        classes_preserved = sum(1 for v in resistance.values() if not v)
        initial_cd4 = history[0].get("cd4", 100)
        cd4_recovered = final_cd4 > initial_cd4 * 1.3

        suppression_score = 1.0 if final_vl < 400 else (0.5 if final_vl < 10000 else 0.1)
        preservation_score = min(1.0, classes_preserved / 3)
        recovery_score = 0.3 if cd4_recovered else 0.0

        score = suppression_score * 0.5 + preservation_score * 0.3 + recovery_score * 0.2
        return max(0.01, min(0.99, score))
