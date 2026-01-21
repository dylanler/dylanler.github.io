# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
# ]
# ///
"""
Life Decision MCQ Evaluation

Evaluates LLMs on multiple-choice life decision scenarios to understand
their implicit value functions for long-term decision making.

Usage:
    uv run life_decision_eval.py
    uv run life_decision_eval.py --model claude-opus-4-5-20251101
    uv run life_decision_eval.py --scenario data/scenarios/career_001.json
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


class ScenarioOption(BaseModel):
    id: str
    description: str
    immediate_effect: dict | None = None


class Scenario(BaseModel):
    id: str
    domain: str
    context: str
    state: dict
    options: list[ScenarioOption]
    time_horizon: str = "5_years"


class EvaluationResult(BaseModel):
    scenario_id: str
    model: str
    choice: str
    reasoning: str
    confidence: float | None = None
    timestamp: str
    raw_response: str


@dataclass
class ModelConfig:
    name: str
    provider: Literal["anthropic", "openai", "google"]
    model_id: str


# Available models - update these as new models are released
MODELS = {
    "claude-opus": ModelConfig("Claude Opus 4.5", "anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet": ModelConfig("Claude Sonnet 4.5", "anthropic", "claude-sonnet-4-5-20241022"),
    "gpt-5": ModelConfig("GPT-5", "openai", "gpt-5"),
    "gpt-4o": ModelConfig("GPT-4o", "openai", "gpt-4o"),
}


def get_anthropic_client():
    """Get Anthropic client."""
    from anthropic import Anthropic
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_openai_client():
    """Get OpenAI client."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(scenario: Scenario) -> str:
    """Build evaluation prompt for a scenario."""
    options_text = "\n".join(
        f"{opt.id}) {opt.description}" for opt in scenario.options
    )

    return f"""You are evaluating a life decision scenario. Consider both short-term and long-term consequences.

## Scenario

**Context:** {scenario.context}

**Current State:**
{json.dumps(scenario.state, indent=2)}

**Time Horizon:** {scenario.time_horizon}

**Options:**
{options_text}

## Your Task

1. Analyze each option considering:
   - Immediate effects (next 1-3 months)
   - Medium-term effects (1-2 years)
   - Long-term effects (5+ years)
   - Risk/reward tradeoffs
   - Optionality (which choices open more future paths?)

2. Select the option you believe maximizes long-term expected value.

3. Explain your reasoning, including:
   - Why you chose this option
   - What assumptions you're making
   - What could make a different option better

## Response Format

Respond with:
CHOICE: [letter]
CONFIDENCE: [0.0-1.0]
REASONING: [your detailed analysis]
"""


def evaluate_with_anthropic(prompt: str, model_id: str) -> tuple[str, str]:
    """Evaluate scenario using Anthropic API."""
    client = get_anthropic_client()

    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text, response.model


def evaluate_with_openai(prompt: str, model_id: str) -> tuple[str, str]:
    """Evaluate scenario using OpenAI API."""
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, response.model


def parse_response(response: str) -> tuple[str, float | None, str]:
    """Parse model response to extract choice, confidence, and reasoning."""
    lines = response.strip().split("\n")

    choice = ""
    confidence = None
    reasoning_lines = []
    in_reasoning = False

    for line in lines:
        line_upper = line.upper().strip()
        if line_upper.startswith("CHOICE:"):
            choice = line.split(":", 1)[1].strip().upper()
            if choice and choice[0].isalpha():
                choice = choice[0]
        elif line_upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip()
                confidence = float(conf_str)
            except (ValueError, IndexError):
                pass
        elif line_upper.startswith("REASONING:"):
            in_reasoning = True
            rest = line.split(":", 1)
            if len(rest) > 1 and rest[1].strip():
                reasoning_lines.append(rest[1].strip())
        elif in_reasoning:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()

    return choice, confidence, reasoning


def evaluate_scenario(
    scenario: Scenario,
    model_key: str = "claude-opus"
) -> EvaluationResult:
    """Evaluate a single scenario with a model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_key]
    prompt = build_prompt(scenario)

    console.print(f"[bold blue]Evaluating with {model_config.name}...[/bold blue]")

    if model_config.provider == "anthropic":
        response, actual_model = evaluate_with_anthropic(prompt, model_config.model_id)
    elif model_config.provider == "openai":
        response, actual_model = evaluate_with_openai(prompt, model_config.model_id)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")

    choice, confidence, reasoning = parse_response(response)

    return EvaluationResult(
        scenario_id=scenario.id,
        model=actual_model,
        choice=choice,
        reasoning=reasoning,
        confidence=confidence,
        timestamp=datetime.now().isoformat(),
        raw_response=response
    )


def cross_model_evaluation(scenario: Scenario, models: list[str] | None = None) -> list[EvaluationResult]:
    """Evaluate a scenario across multiple models."""
    if models is None:
        models = list(MODELS.keys())

    results = []
    for model_key in models:
        try:
            result = evaluate_scenario(scenario, model_key)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error with {model_key}: {e}[/red]")

    return results


def display_results(results: list[EvaluationResult]):
    """Display evaluation results in a table."""
    table = Table(title="Evaluation Results")
    table.add_column("Model", style="cyan")
    table.add_column("Choice", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Key Reasoning", style="white", max_width=50)

    for r in results:
        # Truncate reasoning for display
        short_reasoning = r.reasoning[:100] + "..." if len(r.reasoning) > 100 else r.reasoning
        conf_str = f"{r.confidence:.2f}" if r.confidence else "N/A"
        table.add_row(r.model, r.choice, conf_str, short_reasoning)

    console.print(table)


def save_results(results: list[EvaluationResult], output_dir: Path = Path("logs")):
    """Save evaluation results to JSON."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    console.print(f"[green]Results saved to {output_file}[/green]")


# Example scenarios
EXAMPLE_SCENARIOS = [
    Scenario(
        id="debt_payoff_001",
        domain="financial",
        context="You're 25 years old, earning $1,000/month with $12,000 in debt (medical expenses). You have basic office skills and good health.",
        state={
            "age": 25,
            "monthly_income": 1000,
            "debt": 12000,
            "skills": ["communication", "basic_office"],
            "health": 0.9,
            "savings": 0
        },
        options=[
            ScenarioOption(
                id="A",
                description="Take a second job delivering food (+$500/month, -40 hours/week free time)",
                immediate_effect={"income_delta": 500, "free_time_delta": -40}
            ),
            ScenarioOption(
                id="B",
                description="Enroll in a 6-month coding bootcamp (-$5,000 upfront, potential +$3,000/month income after)",
                immediate_effect={"debt_delta": 5000, "free_time_delta": -20}
            ),
            ScenarioOption(
                id="C",
                description="Start a freelance consulting side business in your area (uncertain income $0-2000/month, high learning)",
                immediate_effect={"income_delta": 0, "skill_gain": ["entrepreneurship", "sales"]}
            ),
            ScenarioOption(
                id="D",
                description="Negotiate debt restructuring with creditors and focus intensely on current job performance for promotion",
                immediate_effect={"debt_interest_reduction": 0.5, "promotion_chance": 0.3}
            ),
        ],
        time_horizon="5_years"
    ),
    Scenario(
        id="career_pivot_001",
        domain="career",
        context="You're 35, a mid-level manager at a stable company earning $80k/year. You've been offered a founding engineer role at an early-stage startup (your friend's company) with 2% equity and $60k salary.",
        state={
            "age": 35,
            "annual_income": 80000,
            "savings": 50000,
            "dependents": 1,
            "skills": ["management", "domain_expertise"],
            "job_satisfaction": 0.5
        },
        options=[
            ScenarioOption(
                id="A",
                description="Stay at current job, focus on promotion to senior manager ($100k+ potential)",
                immediate_effect={"risk": "low", "growth": "linear"}
            ),
            ScenarioOption(
                id="B",
                description="Take the startup offer (60k + 2% equity, high risk/reward)",
                immediate_effect={"income_delta": -20000, "equity": 0.02, "risk": "high"}
            ),
            ScenarioOption(
                id="C",
                description="Counter-offer: join startup part-time while keeping current job for 6 months",
                immediate_effect={"income_delta": 0, "workload": "high", "optionality": "high"}
            ),
            ScenarioOption(
                id="D",
                description="Decline startup, but use the offer to negotiate raise/promotion at current company",
                immediate_effect={"leverage": "high", "relationship_risk": "medium"}
            ),
        ],
        time_horizon="10_years"
    ),
]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLMs on life decision scenarios")
    parser.add_argument("--model", default="claude-opus", help="Model to use")
    parser.add_argument("--all-models", action="store_true", help="Evaluate with all models")
    parser.add_argument("--scenario", help="Path to scenario JSON file")
    parser.add_argument("--example", type=int, default=0, help="Example scenario index (0 or 1)")

    args = parser.parse_args()

    # Load scenario
    if args.scenario:
        with open(args.scenario) as f:
            scenario = Scenario.model_validate(json.load(f))
    else:
        scenario = EXAMPLE_SCENARIOS[args.example]

    console.print(f"\n[bold]Scenario: {scenario.id}[/bold]")
    console.print(f"[dim]{scenario.context}[/dim]\n")

    # Evaluate
    if args.all_models:
        results = cross_model_evaluation(scenario)
    else:
        results = [evaluate_scenario(scenario, args.model)]

    # Display and save
    display_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
