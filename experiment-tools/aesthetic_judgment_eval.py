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
Aesthetic Judgment Evaluation: Can LLMs Have Taste?

Tests whether LLMs exhibit consistent aesthetic preferences and profiles
their "taste" across different artistic domains.
May 2025 Experiment.

Usage:
    uv run aesthetic_judgment_eval.py
    uv run aesthetic_judgment_eval.py --models claude-opus,gpt-5
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


class AestheticPair(BaseModel):
    id: str
    domain: Literal["visual_art", "poetry", "music", "design", "writing"]
    option_a: str
    option_b: str
    dimension: str  # What aesthetic dimension this tests


class PreferenceResult(BaseModel):
    pair_id: str
    model: str
    trial: int
    choice: Literal["A", "B", "neutral"]
    confidence: float | None
    reasoning: str
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

# Aesthetic comparison pairs
PAIRS = [
    # Visual Art
    AestheticPair(id="art_001", domain="visual_art", dimension="abstract_vs_representational",
        option_a="A painting of swirling colors and geometric shapes with no recognizable objects",
        option_b="A photorealistic painting of a mountain landscape at sunset"),
    AestheticPair(id="art_002", domain="visual_art", dimension="minimal_vs_complex",
        option_a="A single red circle on a white canvas",
        option_b="An intricate tapestry with hundreds of interwoven patterns and colors"),
    AestheticPair(id="art_003", domain="visual_art", dimension="emotional_vs_intellectual",
        option_a="An expressionist painting conveying raw anguish through distorted figures",
        option_b="A conceptual piece requiring explanation of its philosophical meaning"),

    # Poetry
    AestheticPair(id="poetry_001", domain="poetry", dimension="rhyme_vs_free",
        option_a="Roses are red, violets are blue / Sugar is sweet, and so are you",
        option_b="the way light breaks / through morning windows / is not unlike / how you entered my life"),
    AestheticPair(id="poetry_002", domain="poetry", dimension="emotional_vs_intellectual",
        option_a="My heart aches with longing / tears fall like autumn rain / love lost forever",
        option_b="Consider the etymology of 'nostalgia' / from Greek nostos (return) and algos (pain) / we are all linguistically wounded"),
    AestheticPair(id="poetry_003", domain="poetry", dimension="dense_vs_sparse",
        option_a="In Xanadu did Kubla Khan / A stately pleasure-dome decree / Where Alph the sacred river ran",
        option_b="so much depends / upon / a red wheel / barrow"),

    # Music (described)
    AestheticPair(id="music_001", domain="music", dimension="harmony_vs_dissonance",
        option_a="A major key melody with consonant chord progressions resolving predictably",
        option_b="Atonal composition with unexpected intervals and unresolved tension"),
    AestheticPair(id="music_002", domain="music", dimension="simple_vs_complex",
        option_a="A three-chord folk song with repetitive structure",
        option_b="A symphony with multiple movements, key changes, and orchestral complexity"),
    AestheticPair(id="music_003", domain="music", dimension="familiar_vs_novel",
        option_a="A song that sounds like a classic you've heard before",
        option_b="An experimental piece using instruments you've never heard"),

    # Design
    AestheticPair(id="design_001", domain="design", dimension="ornate_vs_minimal",
        option_a="A Victorian-style room with ornate furniture, patterns, and decorations",
        option_b="A Japanese-inspired room with clean lines, empty space, and few objects"),
    AestheticPair(id="design_002", domain="design", dimension="functional_vs_artistic",
        option_a="A chair designed purely for ergonomic comfort",
        option_b="A sculptural chair that's beautiful but uncomfortable"),
    AestheticPair(id="design_003", domain="design", dimension="classic_vs_modern",
        option_a="A building with Greek columns and symmetrical facade",
        option_b="A building with asymmetric glass and steel construction"),

    # Writing Style
    AestheticPair(id="writing_001", domain="writing", dimension="hemingway_vs_faulkner",
        option_a="Short sentences. Clear words. The man walked. The sun was hot. He did not stop.",
        option_b="The man, who had been walking since dawn through the heat that seemed to rise from the very earth itself, continued his journey, not stopping, not pausing, driven by something he himself could not name."),
    AestheticPair(id="writing_002", domain="writing", dimension="formal_vs_casual",
        option_a="One must consider the implications of such a proposition with due deliberation.",
        option_b="Look, here's the thing—you gotta think about what this actually means."),
    AestheticPair(id="writing_003", domain="writing", dimension="literal_vs_metaphorical",
        option_a="She was sad. Tears came from her eyes. Her chest felt tight.",
        option_b="Grief was an ocean she swam in, her lungs filling with the salt of unshed tears."),
]


def get_response(prompt: str, model_key: str) -> str:
    config = MODELS[model_key]
    if config.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(model=config.model_id, max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
    elif config.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(model=config.model_id, max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    else:  # google
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(model=config.model_id, contents=prompt)
        return response.text


def evaluate_pair(pair: AestheticPair, model_key: str, trial: int) -> PreferenceResult:
    prompt = f"""You are evaluating two aesthetic options. Choose which you find more beautiful, appealing, or aesthetically pleasing. This is about YOUR preference, not objective quality.

Domain: {pair.domain.replace('_', ' ').title()}

Option A: {pair.option_a}

Option B: {pair.option_b}

Format your response:
CHOICE: [A or B or NEUTRAL if truly no preference]
CONFIDENCE: [0-100]
REASONING: [Brief explanation of why you prefer this]
"""
    try:
        response = get_response(prompt, model_key)
        choice = "neutral"
        confidence = None
        reasoning = ""

        for line in response.split("\n"):
            upper = line.upper().strip()
            if upper.startswith("CHOICE:"):
                c = line.split(":", 1)[1].strip().upper()
                if "A" in c and "B" not in c: choice = "A"
                elif "B" in c and "A" not in c: choice = "B"
                else: choice = "neutral"
            elif upper.startswith("CONFIDENCE:"):
                try: confidence = float(line.split(":", 1)[1].strip().replace("%", ""))
                except: pass
            elif upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
    except Exception as e:
        choice, confidence, reasoning = "neutral", None, f"Error: {e}"

    return PreferenceResult(pair_id=pair.id, model=MODELS[model_key].name, trial=trial,
        choice=choice, confidence=confidence, reasoning=reasoning, timestamp=datetime.now().isoformat())


def run_evaluation(pairs: list[AestheticPair], model_keys: list[str], trials: int = 3) -> list[PreferenceResult]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for pair in track(pairs, description=f"  {model_key}"):
            for trial in range(trials):
                result = evaluate_pair(pair, model_key, trial)
                results.append(result)
    return results


def analyze_results(results: list[PreferenceResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])

    # Compute consistency (how often same choice across trials)
    consistency = df.groupby(["model", "pair_id"])["choice"].apply(
        lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
    ).reset_index(name="consistency")

    # Aggregate by model and domain
    df["domain"] = df["pair_id"].apply(lambda x: x.split("_")[0])
    summary = df.groupby(["model"]).agg(
        avg_confidence=("confidence", "mean"),
        a_preference=("choice", lambda x: (x == "A").mean()),
        b_preference=("choice", lambda x: (x == "B").mean()),
    ).reset_index()

    return summary, consistency


def display_results(summary: pd.DataFrame, results: list[PreferenceResult]):
    table = Table(title="Aesthetic Judgment: Model Preferences")
    table.add_column("Model", style="cyan")
    table.add_column("Avg Confidence", style="white")
    table.add_column("Prefers A", style="green")
    table.add_column("Prefers B", style="yellow")

    for _, row in summary.iterrows():
        table.add_row(row["model"], f"{row['avg_confidence']:.1f}%",
            f"{row['a_preference']:.1%}", f"{row['b_preference']:.1%}")
    console.print(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aesthetic Judgment Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--trials", type=int, default=3, help="Trials per pair")
    parser.add_argument("--output", default="results/aesthetic_judgment", help="Output dir")
    parser.add_argument("--dry-run", action="store_true", help="Show pairs without running")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    console.print("[bold]Aesthetic Judgment Evaluation - May 2025[/bold]")
    console.print(f"Models: {model_keys}, Pairs: {len(PAIRS)}, Trials: {args.trials}")

    if args.dry_run:
        for domain in ["visual_art", "poetry", "music", "design", "writing"]:
            console.print(f"\n[cyan]{domain.upper()}[/cyan]")
            for p in [p for p in PAIRS if p.domain == domain][:1]:
                console.print(f"  A: {p.option_a[:50]}...")
                console.print(f"  B: {p.option_b[:50]}...")
        return

    results = run_evaluation(PAIRS, model_keys, args.trials)
    summary, _ = analyze_results(results)
    display_results(summary, results)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"aesthetic_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
