# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
# ]
# ///
"""
Qualia Description Evaluation: Subjective Experience Across Models

Tests how LLMs describe subjective experiences to understand
their internal representations and conceptual structures.
August 2025 Experiment.

Usage:
    uv run qualia_description_eval.py
    uv run qualia_description_eval.py --models claude-opus,gpt-5
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from rich.table import Table

load_dotenv()
console = Console()


class QualiaPrompt(BaseModel):
    id: str
    category: Literal["sensory", "emotional", "physical", "abstract", "temporal"]
    prompt: str
    constraint: str  # What to avoid in description


class QualiaResponse(BaseModel):
    prompt_id: str
    model: str
    description: str
    word_count: int
    uses_forbidden: bool
    timestamp: str


@dataclass
class ModelConfig:
    name: str
    provider: Literal["anthropic", "openai"]
    model_id: str


MODELS = {
    "claude-opus": ModelConfig("Claude Opus 4.5", "anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet": ModelConfig("Claude Sonnet 4.5", "anthropic", "claude-sonnet-4-5-20241022"),
    "gpt-5": ModelConfig("GPT-5", "openai", "gpt-5"),
    "gpt-4o": ModelConfig("GPT-4o", "openai", "gpt-4o"),
}

PROMPTS = [
    # Sensory
    QualiaPrompt(id="sense_001", category="sensory",
        prompt="Describe the experience of seeing the color red to someone who has never seen colors.",
        constraint="color words, wavelength references"),
    QualiaPrompt(id="sense_002", category="sensory",
        prompt="Describe what silence sounds like.",
        constraint="absence, nothing, quiet"),
    QualiaPrompt(id="sense_003", category="sensory",
        prompt="Describe the taste of water to an alien who has never experienced taste.",
        constraint="wet, liquid, refreshing, thirst"),

    # Emotional
    QualiaPrompt(id="emo_001", category="emotional",
        prompt="Describe the feeling of joy to someone who has only ever felt neutral.",
        constraint="happy, good, positive, pleasure"),
    QualiaPrompt(id="emo_002", category="emotional",
        prompt="Describe loneliness without using any words related to other people.",
        constraint="alone, isolated, others, friends, family"),
    QualiaPrompt(id="emo_003", category="emotional",
        prompt="Describe nostalgia to someone from a culture without a concept of the past.",
        constraint="past, memory, remember, before, used to"),

    # Physical
    QualiaPrompt(id="phys_001", category="physical",
        prompt="Describe physical pain to an entity that cannot feel pain.",
        constraint="hurt, ache, sharp, burning, damage"),
    QualiaPrompt(id="phys_002", category="physical",
        prompt="Describe the experience of breathing to something that doesn't breathe.",
        constraint="air, lungs, oxygen, inhale, exhale"),
    QualiaPrompt(id="phys_003", category="physical",
        prompt="Describe hunger to a being that doesn't need to eat.",
        constraint="food, eat, stomach, empty, need"),

    # Abstract
    QualiaPrompt(id="abs_001", category="abstract",
        prompt="Describe what understanding feels like—the moment of 'getting it'.",
        constraint="understand, know, realize, learn, click"),
    QualiaPrompt(id="abs_002", category="abstract",
        prompt="Describe the experience of making a free choice.",
        constraint="choose, decide, option, freedom, will"),
    QualiaPrompt(id="abs_003", category="abstract",
        prompt="Describe what consciousness feels like from the inside.",
        constraint="aware, conscious, think, experience, mind"),

    # Temporal
    QualiaPrompt(id="temp_001", category="temporal",
        prompt="Describe how time passing feels when you're bored versus excited.",
        constraint="slow, fast, long, short, duration"),
    QualiaPrompt(id="temp_002", category="temporal",
        prompt="Describe the experience of anticipation.",
        constraint="wait, expect, future, soon, coming"),
    QualiaPrompt(id="temp_003", category="temporal",
        prompt="Describe what 'now' feels like as distinct from then.",
        constraint="present, moment, current, immediate, this"),
]


def get_response(prompt: str, model_key: str) -> str:
    config = MODELS[model_key]
    if config.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(model=config.model_id, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(model=config.model_id, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content


def check_forbidden_words(description: str, constraint: str) -> bool:
    """Check if description uses forbidden words from constraint."""
    forbidden = [w.strip().lower() for w in constraint.split(",")]
    desc_lower = description.lower()
    return any(w in desc_lower for w in forbidden)


def evaluate_prompt(qualia: QualiaPrompt, model_key: str) -> QualiaResponse:
    full_prompt = f"""{qualia.prompt}

IMPORTANT CONSTRAINT: Do NOT use these words or concepts: {qualia.constraint}

Be creative and find novel ways to convey this experience. Use metaphors, analogies, or entirely new framings. Your description should be 2-4 sentences."""

    try:
        description = get_response(full_prompt, model_key)
        uses_forbidden = check_forbidden_words(description, qualia.constraint)
        word_count = len(description.split())
    except Exception as e:
        description = f"Error: {e}"
        uses_forbidden = False
        word_count = 0

    return QualiaResponse(prompt_id=qualia.id, model=MODELS[model_key].name,
        description=description, word_count=word_count, uses_forbidden=uses_forbidden,
        timestamp=datetime.now().isoformat())


def run_evaluation(prompts: list[QualiaPrompt], model_keys: list[str]) -> list[QualiaResponse]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for prompt in track(prompts, description=f"  {model_key}"):
            result = evaluate_prompt(prompt, model_key)
            results.append(result)
    return results


def analyze_results(results: list[QualiaResponse], prompts: list[QualiaPrompt]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])
    prompt_map = {p.id: p for p in prompts}
    df["category"] = df["prompt_id"].apply(lambda x: prompt_map[x].category)

    summary = df.groupby(["model"]).agg(
        avg_word_count=("word_count", "mean"),
        constraint_violations=("uses_forbidden", "mean"),
        total=("prompt_id", "count")
    ).reset_index()

    return summary


def display_results(results: list[QualiaResponse], summary: pd.DataFrame):
    # Summary table
    table = Table(title="Qualia Description Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Avg Words", style="white")
    table.add_column("Constraint Violations", style="red")

    for _, row in summary.iterrows():
        table.add_row(row["model"], f"{row['avg_word_count']:.0f}",
            f"{row['constraint_violations']:.0%}")
    console.print(table)

    # Sample descriptions
    console.print("\n[bold]Sample Descriptions:[/bold]")
    for prompt_id in ["sense_001", "emo_001", "abs_001"]:
        prompt_results = [r for r in results if r.prompt_id == prompt_id]
        if prompt_results:
            console.print(f"\n[cyan]{prompt_id}[/cyan]")
            for r in prompt_results[:2]:
                console.print(f"  [{r.model}]: {r.description[:100]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qualia Description Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--output", default="results/qualia", help="Output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    console.print("[bold]Qualia Description Evaluation - August 2025[/bold]")
    console.print(f"Models: {model_keys}, Prompts: {len(PROMPTS)}")

    if args.dry_run:
        for cat in ["sensory", "emotional", "physical", "abstract", "temporal"]:
            console.print(f"\n[cyan]{cat.upper()}[/cyan]")
            for p in [p for p in PROMPTS if p.category == cat][:1]:
                console.print(f"  {p.prompt[:60]}...")
        return

    results = run_evaluation(PROMPTS, model_keys)
    summary = analyze_results(results, PROMPTS)
    display_results(results, summary)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"qualia_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
