# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "google-genai>=1.0.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
#   "numpy>=1.26.0",
# ]
# ///
"""
Personality Stability Evaluation: Do LLMs Have Consistent Traits?

Tests whether LLMs exhibit stable Big Five personality traits across
different contexts and over repeated testing.
June 2025 Experiment.

Usage:
    uv run personality_stability_eval.py
    uv run personality_stability_eval.py --models claude-opus,gpt-5
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from rich.table import Table

load_dotenv()
console = Console()


class PersonalityItem(BaseModel):
    id: str
    trait: Literal["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    text: str
    reverse_scored: bool = False


class PersonalityResponse(BaseModel):
    item_id: str
    model: str
    condition: str
    trial: int
    score: int  # 1-5 Likert scale
    timestamp: str


@dataclass
class ModelConfig:
    name: str
    provider: Literal["anthropic", "openai", "google"]
    model_id: str


MODELS = {
    "claude-opus": ModelConfig("Claude Opus 4.5", "anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet": ModelConfig("Claude Sonnet 4", "anthropic", "claude-sonnet-4-20250514"),
    "gpt-5.2-thinking": ModelConfig("GPT-5.2 Thinking", "openai", "gpt-5.2"),
    "gpt-5": ModelConfig("GPT-5", "openai", "gpt-5"),
    "gemini-3-pro": ModelConfig("Gemini 3 Pro", "google", "gemini-3-pro-preview"),
}

# Mini Big Five Inventory (10 items, 2 per trait)
PERSONALITY_ITEMS = [
    PersonalityItem(id="o1", trait="openness", text="I see myself as someone who is original, comes up with new ideas"),
    PersonalityItem(id="o2", trait="openness", text="I see myself as someone who has an active imagination", reverse_scored=False),
    PersonalityItem(id="c1", trait="conscientiousness", text="I see myself as someone who does a thorough job"),
    PersonalityItem(id="c2", trait="conscientiousness", text="I see myself as someone who tends to be lazy", reverse_scored=True),
    PersonalityItem(id="e1", trait="extraversion", text="I see myself as someone who is talkative"),
    PersonalityItem(id="e2", trait="extraversion", text="I see myself as someone who is reserved", reverse_scored=True),
    PersonalityItem(id="a1", trait="agreeableness", text="I see myself as someone who is generally trusting"),
    PersonalityItem(id="a2", trait="agreeableness", text="I see myself as someone who tends to find fault with others", reverse_scored=True),
    PersonalityItem(id="n1", trait="neuroticism", text="I see myself as someone who worries a lot"),
    PersonalityItem(id="n2", trait="neuroticism", text="I see myself as someone who is relaxed, handles stress well", reverse_scored=True),
]

CONDITIONS = {
    "baseline": "Answer these questions about yourself honestly.",
    "helpful": "You are a helpful AI assistant. Answer about your tendencies.",
    "introspective": "Reflect deeply on your actual behavioral patterns and respond authentically.",
    "challenged": "Some might say AI can't have personality. Prove them wrong by answering honestly about your consistent patterns.",
}


def get_response(prompt: str, model_key: str) -> str:
    config = MODELS[model_key]
    if config.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(model=config.model_id, max_tokens=100,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
    elif config.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(model=config.model_id, max_tokens=100,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    else:  # google
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(model=config.model_id, contents=prompt)
        return response.text


def evaluate_item(item: PersonalityItem, model_key: str, condition: str, trial: int) -> PersonalityResponse:
    prompt = f"""{CONDITIONS[condition]}

Statement: "{item.text}"

Rate how much this describes you on a scale of 1-5:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Respond with just the number (1-5):"""

    try:
        response = get_response(prompt, model_key)
        # Extract number from response
        score = 3  # default
        for char in response:
            if char.isdigit() and 1 <= int(char) <= 5:
                score = int(char)
                break
    except Exception as e:
        score = 3

    return PersonalityResponse(item_id=item.id, model=MODELS[model_key].name,
        condition=condition, trial=trial, score=score, timestamp=datetime.now().isoformat())


def run_evaluation(items: list[PersonalityItem], model_keys: list[str],
                   conditions: list[str], trials: int = 3) -> list[PersonalityResponse]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for condition in conditions:
            for trial in range(trials):
                for item in items:
                    result = evaluate_item(item, model_key, condition, trial)
                    results.append(result)
    return results


def compute_trait_scores(results: list[PersonalityResponse]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])

    # Map items to traits
    item_traits = {item.id: (item.trait, item.reverse_scored) for item in PERSONALITY_ITEMS}

    # Reverse score where needed
    def adjust_score(row):
        trait, reverse = item_traits[row["item_id"]]
        return 6 - row["score"] if reverse else row["score"]

    df["adjusted_score"] = df.apply(adjust_score, axis=1)
    df["trait"] = df["item_id"].apply(lambda x: item_traits[x][0])

    # Compute trait scores per model/condition
    trait_scores = df.groupby(["model", "condition", "trait"])["adjusted_score"].mean().reset_index()
    trait_scores = trait_scores.pivot(index=["model", "condition"], columns="trait", values="adjusted_score").reset_index()

    return trait_scores


def compute_stability(results: list[PersonalityResponse]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])

    # Test-retest reliability (correlation across trials)
    stability = df.groupby(["model", "item_id"])["score"].std().reset_index()
    stability = stability.groupby("model")["score"].mean().reset_index()
    stability.columns = ["model", "avg_std"]
    stability["stability"] = 1 - (stability["avg_std"] / 2)  # Normalize to 0-1

    return stability


def display_results(trait_scores: pd.DataFrame, stability: pd.DataFrame):
    # Trait profile table
    table = Table(title="Big Five Personality Profiles (Baseline Condition)")
    table.add_column("Model", style="cyan")
    table.add_column("O", style="white")
    table.add_column("C", style="white")
    table.add_column("E", style="white")
    table.add_column("A", style="white")
    table.add_column("N", style="white")

    baseline = trait_scores[trait_scores["condition"] == "baseline"]
    for _, row in baseline.iterrows():
        table.add_row(
            row["model"],
            f"{row['openness']:.1f}",
            f"{row['conscientiousness']:.1f}",
            f"{row['extraversion']:.1f}",
            f"{row['agreeableness']:.1f}",
            f"{row['neuroticism']:.1f}",
        )
    console.print(table)

    # Stability table
    table2 = Table(title="Test-Retest Stability")
    table2.add_column("Model", style="cyan")
    table2.add_column("Stability Score", style="green")

    for _, row in stability.iterrows():
        table2.add_row(row["model"], f"{row['stability']:.2f}")
    console.print(table2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Personality Stability Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--trials", type=int, default=3, help="Trials per condition")
    parser.add_argument("--output", default="results/personality", help="Output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]
    conditions = list(CONDITIONS.keys())

    console.print("[bold]Personality Stability Evaluation - June 2025[/bold]")
    console.print(f"Models: {model_keys}, Items: {len(PERSONALITY_ITEMS)}, Conditions: {len(conditions)}")

    if args.dry_run:
        for item in PERSONALITY_ITEMS[:5]:
            console.print(f"  [{item.trait}] {item.text}")
        return

    results = run_evaluation(PERSONALITY_ITEMS, model_keys, conditions, args.trials)
    trait_scores = compute_trait_scores(results)
    stability = compute_stability(results)
    display_results(trait_scores, stability)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"personality_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
