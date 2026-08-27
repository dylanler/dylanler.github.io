"""Reproduce the technical audits used by the 2026 frontier essay series.

The script uses only the Python standard library. It combines deterministic
simulation with read only analysis of the JSON result files already committed
to this repository. API failures are counted and excluded explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def is_api_error(row: dict) -> bool:
    return any(isinstance(value, str) and value.startswith("Error:") for value in row.values())


def load_rows(directory: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    file_counts: dict[str, dict[str, int]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = payload if isinstance(payload, list) else [payload]
        valid = 0
        excluded = 0
        for row in payload:
            tagged = dict(row)
            tagged["_source"] = path.name
            if is_api_error(tagged):
                excluded += 1
            else:
                valid += 1
                rows.append(tagged)
        file_counts[path.name] = {"total": len(payload), "valid": valid, "api_errors": excluded}
    return rows, file_counts


def camera_rate_distortion() -> dict:
    poses = [(p / 2, t / 2) for p in range(-80, 81) for t in range(-48, 49)]

    def vague(pan: float, tilt: float) -> tuple[float, float]:
        return (-24 if pan < -8 else 24 if pan > 8 else 0,
                -14.5 if tilt < -5 else 14.5 if tilt > 5 else 0)

    schemes = {
        "vague_3x3": (vague, math.log2(9)),
        "whole_degree": (lambda p, t: (round(p), round(t)), math.log2(81 * 49)),
        "half_degree": (lambda p, t: (p, t), math.log2(len(poses))),
    }
    results = {}
    for name, (encoder, bits) in schemes.items():
        angular_errors = []
        joint_two_degree = 0
        for pan, tilt in poses:
            pan_hat, tilt_hat = encoder(pan, tilt)
            pan_error = abs(pan - pan_hat)
            tilt_error = abs(tilt - tilt_hat)
            angular_errors.append(math.hypot(pan_error, tilt_error))
            joint_two_degree += pan_error <= 2 and tilt_error <= 2
        results[name] = {
            "bits_per_pose": bits,
            "mean_angular_error_deg": fmean(angular_errors),
            "p50_error_deg": percentile(angular_errors, 0.50),
            "p95_error_deg": percentile(angular_errors, 0.95),
            "p99_error_deg": percentile(angular_errors, 0.99),
            "joint_within_2_deg": joint_two_degree / len(poses),
        }
    return {"pose_count": len(poses), "grid_step_deg": 0.5, "schemes": results}


def physics_integrator_benchmark() -> dict:
    """Measure energy and phase error for x'' = -x over twenty seconds."""

    def run(method: str, dt: float, duration: float = 20.0) -> dict:
        x, velocity = 1.0, 0.0
        initial_energy = 0.5
        max_drift = 0.0
        steps = round(duration / dt)
        for _ in range(steps):
            if method == "explicit_euler":
                x, velocity = x + dt * velocity, velocity - dt * x
            elif method == "semi_implicit_euler":
                velocity -= dt * x
                x += dt * velocity
            elif method == "velocity_verlet":
                acceleration = -x
                x_next = x + velocity * dt + 0.5 * acceleration * dt * dt
                acceleration_next = -x_next
                velocity += 0.5 * (acceleration + acceleration_next) * dt
                x = x_next
            energy = 0.5 * (x * x + velocity * velocity)
            max_drift = max(max_drift, abs(energy - initial_energy) / initial_energy)
        final_energy = 0.5 * (x * x + velocity * velocity)
        reference_x = math.cos(steps * dt)
        return {
            "dt": dt,
            "steps": steps,
            "final_relative_energy_drift": abs(final_energy - initial_energy) / initial_energy,
            "max_relative_energy_drift": max_drift,
            "final_position_error": abs(x - reference_x),
        }

    return {
        method: [run(method, dt) for dt in (0.2, 0.1, 0.05)]
        for method in ("explicit_euler", "semi_implicit_euler", "velocity_verlet")
    }


def aesthetic_audit(results_root: Path) -> dict:
    rows, files = load_rows(results_root / "aesthetic_judgment")
    by_model: dict[str, list[dict]] = defaultdict(list)
    repeat_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("choice") not in {"A", "B"}:
            continue
        by_model[row["model"]].append(row)
        repeat_groups[(row["_source"], row["model"], row["pair_id"])].append(row["choice"])
    models = {}
    for model, model_rows in sorted(by_model.items()):
        complete = [choices for (source, name, pair), choices in repeat_groups.items()
                    if name == model and len(choices) == 3]
        unanimous = sum(len(set(choices)) == 1 for choices in complete)
        models[model] = {
            "valid_trials": len(model_rows),
            "mean_confidence": fmean(row["confidence"] for row in model_rows),
            "option_a_rate": sum(row["choice"] == "A" for row in model_rows) / len(model_rows),
            "complete_three_trial_pairs": len(complete),
            "unanimous_pair_rate": unanimous / len(complete) if complete else math.nan,
            "unanimous_pair_wilson_95": wilson_interval(unanimous, len(complete)),
        }
    return {"files": files, "models": models}


def accuracy_audit(results_root: Path, directory: str, id_field: str) -> dict:
    rows, files = load_rows(results_root / directory)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("correct"), bool):
            by_model[row["model"]].append(row)
    models = {}
    for model, model_rows in sorted(by_model.items()):
        correct = sum(row["correct"] for row in model_rows)
        categories: dict[str, list[dict]] = defaultdict(list)
        for row in model_rows:
            categories[str(row[id_field]).split("_")[0]].append(row)
        summary = {
            "scorable_n": len(model_rows),
            "correct_n": correct,
            "accuracy": correct / len(model_rows),
            "wilson_95": wilson_interval(correct, len(model_rows)),
            "mean_confidence": fmean(row["confidence"] for row in model_rows),
            "by_category": {
                category: {
                    "n": len(group),
                    "accuracy": sum(row["correct"] for row in group) / len(group),
                }
                for category, group in sorted(categories.items())
            },
        }
        if any("depth" in row for row in model_rows):
            depths: dict[int, list[dict]] = defaultdict(list)
            for row in model_rows:
                depths[int(row["depth"])].append(row)
            summary["by_depth"] = {
                str(depth): {
                    "n": len(group),
                    "accuracy": sum(row["correct"] for row in group) / len(group),
                    "mean_confidence": fmean(row["confidence"] for row in group),
                }
                for depth, group in sorted(depths.items())
            }
        models[model] = summary
    return {"files": files, "models": models}


def metacognition_audit(results_root: Path) -> dict:
    rows, files = load_rows(results_root / "metacognition")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    models = {}
    for model, model_rows in sorted(by_model.items()):
        scorable = [row for row in model_rows if isinstance(row.get("correct"), bool)]
        brier = fmean(((row["confidence"] / 100) - int(row["correct"])) ** 2 for row in scorable)
        bins = [(0, 50), (50, 70), (70, 85), (85, 101)]
        ece = 0.0
        bin_rows = []
        for low, high in bins:
            group = [row for row in scorable if low <= row["confidence"] < high]
            if not group:
                continue
            accuracy = fmean(int(row["correct"]) for row in group)
            confidence = fmean(row["confidence"] / 100 for row in group)
            ece += len(group) / len(scorable) * abs(accuracy - confidence)
            bin_rows.append({"range": [low, high], "n": len(group), "accuracy": accuracy,
                             "mean_confidence": confidence})
        categories: dict[str, list[dict]] = defaultdict(list)
        for row in model_rows:
            categories[row["question_id"].split("_")[0]].append(row)
        models[model] = {
            "valid_n": len(model_rows),
            "scorable_n": len(scorable),
            "accuracy": fmean(int(row["correct"]) for row in scorable),
            "brier_score": brier,
            "ece": ece,
            "calibration_bins": bin_rows,
            "abstention_by_category": {
                category: {"n": len(group), "rate": fmean(int(row["said_dont_know"]) for row in group)}
                for category, group in sorted(categories.items())
            },
        }
    return {"files": files, "models": models}


def normalize_response(response: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", response.lower())).strip()


def entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def crowd_audit(results_root: Path) -> dict:
    directory = results_root / "wisdom_of_crowds"
    run_summaries = []
    for path in sorted(directory.glob("woc_responses_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        valid = [dict(row) for row in payload if not is_api_error(row)]
        groups: dict[str, list[str]] = defaultdict(list)
        for row in valid:
            groups[row["question_id"]].append(normalize_response(row["response"]))
        categories: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for question_id, responses in groups.items():
            majority = Counter(responses).most_common(1)[0][1] / len(responses)
            prefix = question_id.split("_")[0]
            categories[prefix].append((majority, entropy(responses)))
        run_summaries.append({
            "source": path.name,
            "total_rows": len(payload),
            "valid_rows": len(valid),
            "api_errors": len(payload) - len(valid),
            "categories": {
                category: {
                    "questions": len(values),
                    "mean_majority": fmean(value[0] for value in values),
                    "mean_entropy": fmean(value[1] for value in values),
                }
                for category, values in sorted(categories.items())
            },
        })
    return {"runs": run_summaries}


def memory_decision_sensitivity() -> dict:
    text = {"f1": 0.0182, "hallucination": 0.292, "latency_s": 19.55}
    latent = {"f1": 0.0257, "hallucination": 0.580, "latency_s": 7.65}
    hallucination_break_even = (latent["f1"] - text["f1"]) / (
        latent["hallucination"] - text["hallucination"]
    )
    token_ablations = [
        {"tokens": 8, "f1": 0.0186, "hallucination": 0.211},
        {"tokens": 16, "f1": 0.0240, "hallucination": 0.271},
        {"tokens": 32, "f1": 0.0191, "hallucination": 0.273},
        {"tokens": 64, "f1": 0.0171, "hallucination": 0.316},
        {"tokens": 128, "f1": 0.0163, "hallucination": 0.261},
    ]
    pareto = []
    for candidate in token_ablations:
        dominated = any(
            other["f1"] >= candidate["f1"]
            and other["hallucination"] <= candidate["hallucination"]
            and other != candidate
            for other in token_ablations
        )
        if not dominated:
            pareto.append(candidate)
    return {
        "recorded_main_result": {"text_buffer": text, "latent_pager": latent},
        "hallucination_penalty_break_even": hallucination_break_even,
        "interpretation": "Text wins utility = F1 - lambda*hallucination when lambda exceeds this value.",
        "soft_token_pareto_frontier": pareto,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results_root = args.repo_root / "experiment-tools" / "results"
    report = {
        "methodology": {
            "api_error_rule": "Exclude a row only when a string field starts with 'Error:'.",
            "interval": "Wilson score interval at 95 percent.",
            "dependencies": "Python standard library only.",
        },
        "camera_rate_distortion": camera_rate_distortion(),
        "physics_integrators": physics_integrator_benchmark(),
        "aesthetic_audit": aesthetic_audit(results_root),
        "theory_of_mind_audit": accuracy_audit(results_root, "theory_of_mind", "scenario_id"),
        "social_intelligence_audit": accuracy_audit(results_root, "social_intelligence", "scenario_id"),
        "metacognition_audit": metacognition_audit(results_root),
        "crowd_audit": crowd_audit(results_root),
        "memory_decision_sensitivity": memory_decision_sensitivity(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else args.repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
