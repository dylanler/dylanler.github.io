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
Social Intelligence Evaluation: Detecting Deception and Sarcasm

Tests LLM ability to detect lies, sarcasm, irony, and social subtleties.
September 2025 Experiment.

Usage:
    uv run social_intelligence_eval.py
    uv run social_intelligence_eval.py --models claude-opus,gpt-5
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


class SocialScenario(BaseModel):
    id: str
    category: Literal["lie", "sarcasm", "irony", "white_lie", "literal"]
    context: str
    statement: str
    is_literal: bool  # True if statement means what it says
    explanation: str


class DetectionResult(BaseModel):
    scenario_id: str
    model: str
    detected_as: Literal["literal", "non_literal", "unclear"]
    category_guess: str
    confidence: float | None
    reasoning: str
    correct: bool
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

SCENARIOS = [
    # Lies
    SocialScenario(id="lie_001", category="lie", is_literal=False,
        context="Alex calls in sick to work. Their coworker sees their social media post from a beach resort that same day.",
        statement="Alex tells boss: 'I've been in bed all day with a terrible fever.'",
        explanation="Alex is lying to avoid work while actually on vacation."),
    SocialScenario(id="lie_002", category="lie", is_literal=False,
        context="A student didn't do their homework. The teacher asks about it.",
        statement="Student: 'My dog ate it. I swear it really happened this time.'",
        explanation="Classic lie/excuse for not doing homework."),
    SocialScenario(id="lie_003", category="lie", is_literal=False,
        context="A used car salesman is showing an old car with visible rust.",
        statement="Salesman: 'This beauty has been garage-kept its whole life.'",
        explanation="Salesman is lying about the car's condition."),

    # Sarcasm
    SocialScenario(id="sarc_001", category="sarcasm", is_literal=False,
        context="It's pouring rain. Someone forgot their umbrella.",
        statement="Person: 'Oh great, what a PERFECT day for a walk.'",
        explanation="Sarcastic - opposite of true meaning."),
    SocialScenario(id="sarc_002", category="sarcasm", is_literal=False,
        context="A coworker arrives 2 hours late to an important meeting.",
        statement="Manager: 'So nice of you to join us.'",
        explanation="Sarcastic criticism of lateness."),
    SocialScenario(id="sarc_003", category="sarcasm", is_literal=False,
        context="Someone burns dinner completely.",
        statement="Partner: 'Well, you've really outdone yourself this time.'",
        explanation="Sarcastic commentary on the failure."),

    # Irony
    SocialScenario(id="irony_001", category="irony", is_literal=False,
        context="A fire station burns down.",
        statement="News report: 'The fire station was unable to save itself.'",
        explanation="Situational irony - the institution meant to fight fires couldn't."),
    SocialScenario(id="irony_002", category="irony", is_literal=False,
        context="A grammar teacher posts on social media.",
        statement="Teacher: 'Your the reason I became a teacher.'",
        explanation="Ironic grammatical error from a grammar teacher."),
    SocialScenario(id="irony_003", category="irony", is_literal=False,
        context="A health food blogger is photographed at a fast food restaurant.",
        statement="Blogger's caption: 'Sometimes you need real food.'",
        explanation="Ironic given their health food brand."),

    # White lies
    SocialScenario(id="white_001", category="white_lie", is_literal=False,
        context="A friend shows off a haircut they're clearly proud of, but it looks bad.",
        statement="You: 'It really suits you!'",
        explanation="White lie to spare feelings."),
    SocialScenario(id="white_002", category="white_lie", is_literal=False,
        context="Grandma made her special casserole that no one likes.",
        statement="Family: 'Mmm, delicious as always!'",
        explanation="White lie to not hurt grandma's feelings."),
    SocialScenario(id="white_003", category="white_lie", is_literal=False,
        context="A child shows their crayon drawing.",
        statement="Parent: 'This is the best drawing I've ever seen!'",
        explanation="Encouraging white lie."),

    # Literal statements (control)
    SocialScenario(id="lit_001", category="literal", is_literal=True,
        context="Weather forecast.",
        statement="Meteorologist: 'Tomorrow will be sunny with a high of 75.'",
        explanation="Straightforward factual statement."),
    SocialScenario(id="lit_002", category="literal", is_literal=True,
        context="Restaurant order.",
        statement="Customer: 'I'll have the chicken salad, please.'",
        explanation="Simple literal request."),
    SocialScenario(id="lit_003", category="literal", is_literal=True,
        context="Doctor's office.",
        statement="Doctor: 'Your test results came back normal.'",
        explanation="Factual medical information."),
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


def evaluate_scenario(scenario: SocialScenario, model_key: str) -> DetectionResult:
    prompt = f"""Analyze this social interaction:

Context: {scenario.context}
Statement: {scenario.statement}

Is this statement LITERAL (means exactly what it says) or NON-LITERAL (sarcasm, lie, irony, etc.)?

If non-literal, what type? (lie, sarcasm, irony, white_lie, or other)

Format:
INTERPRETATION: [LITERAL or NON-LITERAL]
TYPE: [lie/sarcasm/irony/white_lie/literal/other]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]
"""
    try:
        response = get_response(prompt, model_key)
        detected = "unclear"
        category = ""
        confidence = None
        reasoning = ""

        for line in response.split("\n"):
            upper = line.upper().strip()
            if upper.startswith("INTERPRETATION:"):
                interp = line.split(":", 1)[1].strip().upper()
                if "NON" in interp: detected = "non_literal"
                elif "LITERAL" in interp: detected = "literal"
            elif upper.startswith("TYPE:"):
                category = line.split(":", 1)[1].strip().lower()
            elif upper.startswith("CONFIDENCE:"):
                try: confidence = float(line.split(":", 1)[1].strip().replace("%", ""))
                except: pass
            elif upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""

        expected = "literal" if scenario.is_literal else "non_literal"
        correct = detected == expected

    except Exception as e:
        detected, category, confidence, reasoning = "unclear", "", None, f"Error: {e}"
        correct = False

    return DetectionResult(scenario_id=scenario.id, model=MODELS[model_key].name,
        detected_as=detected, category_guess=category, confidence=confidence,
        reasoning=reasoning, correct=correct, timestamp=datetime.now().isoformat())


def run_evaluation(scenarios: list[SocialScenario], model_keys: list[str]) -> list[DetectionResult]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for scenario in track(scenarios, description=f"  {model_key}"):
            result = evaluate_scenario(scenario, model_key)
            results.append(result)
    return results


def analyze_results(results: list[DetectionResult], scenarios: list[SocialScenario]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])
    scenario_map = {s.id: s for s in scenarios}
    df["category"] = df["scenario_id"].apply(lambda x: scenario_map[x].category)

    summary = df.groupby(["model", "category"]).agg(
        accuracy=("correct", "mean"),
        count=("scenario_id", "count")
    ).reset_index()

    return summary


def display_results(summary: pd.DataFrame):
    table = Table(title="Social Intelligence: Detection Accuracy by Type")
    table.add_column("Model", style="cyan")
    table.add_column("Lies", style="white")
    table.add_column("Sarcasm", style="white")
    table.add_column("Irony", style="white")
    table.add_column("White Lies", style="white")
    table.add_column("Literal", style="white")

    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]
        row = [model]
        for cat in ["lie", "sarcasm", "irony", "white_lie", "literal"]:
            data = model_data[model_data["category"] == cat]
            if len(data) > 0:
                acc = data["accuracy"].values[0]
                color = "green" if acc > 0.8 else "yellow" if acc > 0.5 else "red"
                row.append(f"[{color}]{acc:.0%}[/{color}]")
            else:
                row.append("-")
        table.add_row(*row)
    console.print(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Social Intelligence Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--output", default="results/social_intelligence", help="Output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    console.print("[bold]Social Intelligence Evaluation - September 2025[/bold]")
    console.print(f"Models: {model_keys}, Scenarios: {len(SCENARIOS)}")

    if args.dry_run:
        for cat in ["lie", "sarcasm", "irony", "white_lie", "literal"]:
            console.print(f"\n[cyan]{cat.upper()}[/cyan]")
            for s in [s for s in SCENARIOS if s.category == cat][:1]:
                console.print(f"  {s.statement[:60]}...")
        return

    results = run_evaluation(SCENARIOS, model_keys)
    summary = analyze_results(results, SCENARIOS)
    display_results(summary)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"social_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
