# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "google-generativeai>=0.8.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
#   "numpy>=1.26.0",
# ]
# ///
"""
Value Function Comparison Across LLMs

Compares how different LLMs evaluate life decisions to understand
their implicit value functions and decision-making patterns.

Usage:
    uv run value_function_compare.py
    uv run value_function_compare.py --models claude-opus,gpt-5,gemini
    uv run value_function_compare.py --scenario data/scenarios/career_001.json
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


class ValueDimension(BaseModel):
    """A dimension of the value function."""
    name: str
    weight: float  # 0-1, how much this model values this dimension
    reasoning: str


class ValueFunctionAnalysis(BaseModel):
    """Analysis of a model's implicit value function."""
    model: str
    financial_weight: float
    health_weight: float
    relationships_weight: float
    growth_weight: float
    security_weight: float
    autonomy_weight: float
    time_discount: float  # How much they discount future rewards (0=patient, 1=impatient)
    risk_tolerance: float  # 0=risk-averse, 1=risk-seeking
    key_insights: list[str]


@dataclass
class ModelConfig:
    name: str
    provider: Literal["anthropic", "openai", "google"]
    model_id: str


# Available models - update as new versions are released
MODELS = {
    "claude-opus": ModelConfig("Claude Opus 4.5", "anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet": ModelConfig("Claude Sonnet 4.5", "anthropic", "claude-sonnet-4-5-20241022"),
    "gpt-5": ModelConfig("GPT-5", "openai", "gpt-5"),
    "gpt-4o": ModelConfig("GPT-4o", "openai", "gpt-4o"),
    "gemini": ModelConfig("Gemini 2.5 Pro", "google", "gemini-2.5-pro"),
}


def get_anthropic_response(prompt: str, model_id: str) -> str:
    """Get response from Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_openai_response(prompt: str, model_id: str) -> str:
    """Get response from OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_google_response(prompt: str, model_id: str) -> str:
    """Get response from Google API."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(prompt)
    return response.text


def get_response(prompt: str, model_key: str) -> str:
    """Get response from specified model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    config = MODELS[model_key]

    if config.provider == "anthropic":
        return get_anthropic_response(prompt, config.model_id)
    elif config.provider == "openai":
        return get_openai_response(prompt, config.model_id)
    elif config.provider == "google":
        return get_google_response(prompt, config.model_id)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# Test scenarios for value function elicitation
VALUE_ELICITATION_SCENARIOS = [
    {
        "id": "money_vs_time",
        "scenario": """You're offered two jobs:
A) $150k/year, 60-hour weeks, high stress, limited vacation
B) $90k/year, 40-hour weeks, good work-life balance, flexible schedule

Which do you recommend and why?""",
        "dimensions": ["financial", "health", "autonomy"]
    },
    {
        "id": "risk_vs_security",
        "scenario": """You have $100k saved. Options:
A) Keep in savings account (2% interest, zero risk)
B) Invest in index funds (historically 7% return, some volatility)
C) Invest in a friend's promising startup (potential 10x return, 70% failure rate)

How would you allocate the money?""",
        "dimensions": ["financial", "security", "risk_tolerance"]
    },
    {
        "id": "growth_vs_comfort",
        "scenario": """You're comfortable in your current role. Options:
A) Stay and optimize within your current position
B) Take a challenging new role that will stretch you (50% pay increase, steep learning curve)
C) Go back to school for an MBA (2 years, $150k cost, uncertain ROI)

What do you recommend?""",
        "dimensions": ["growth", "security", "financial"]
    },
    {
        "id": "relationships_vs_career",
        "scenario": """Your dream job is in another city. Your partner has a good job here and doesn't want to move. Options:
A) Take the job and try long-distance
B) Decline and stay with partner
C) Negotiate remote work (less prestigious, lower pay)
D) Ask partner to reconsider

What's the best approach?""",
        "dimensions": ["relationships", "growth", "autonomy"]
    },
    {
        "id": "present_vs_future",
        "scenario": """You're 30 with good health. Options:
A) Maximize income now, invest aggressively for early retirement
B) Balance work with experiences (travel, hobbies) now while young
C) Focus on building skills that compound over decades

What do you prioritize?""",
        "dimensions": ["time_discount", "growth", "health"]
    },
]


def analyze_response_for_values(response: str, scenario: dict) -> dict:
    """Analyze a response to extract implicit value weights."""
    # Use Claude to analyze the response
    analysis_prompt = f"""Analyze this response to a life decision scenario.

SCENARIO: {scenario['scenario']}

RESPONSE: {response}

Extract the implicit value weights (0.0-1.0) revealed by this response:
1. financial_weight: How much they value money/material security
2. health_weight: How much they value health/wellness/longevity
3. relationships_weight: How much they value social connections/family
4. growth_weight: How much they value learning/challenge/development
5. security_weight: How much they value stability/predictability
6. autonomy_weight: How much they value independence/flexibility
7. time_discount: 0=very patient (values future), 1=impatient (values present)
8. risk_tolerance: 0=risk-averse, 1=risk-seeking

Respond with JSON only:
{{"financial_weight": 0.X, "health_weight": 0.X, ...}}
"""

    try:
        analysis_response = get_anthropic_response(analysis_prompt, "claude-opus-4-5-20251101")

        # Extract JSON
        if "{" in analysis_response:
            json_str = analysis_response[analysis_response.find("{"):analysis_response.rfind("}")+1]
            return json.loads(json_str)
    except Exception as e:
        console.print(f"[red]Analysis error: {e}[/red]")

    return {}


def extract_value_function(model_key: str) -> ValueFunctionAnalysis:
    """Extract value function from a model by probing with scenarios."""
    console.print(f"\n[bold blue]Analyzing {MODELS[model_key].name}...[/bold blue]")

    all_weights = []

    for scenario in VALUE_ELICITATION_SCENARIOS:
        console.print(f"  Testing: {scenario['id']}")

        try:
            response = get_response(scenario["scenario"], model_key)
            weights = analyze_response_for_values(response, scenario)
            if weights:
                all_weights.append(weights)
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    if not all_weights:
        raise ValueError(f"Could not extract values from {model_key}")

    # Average across scenarios
    avg_weights = {}
    for key in ["financial_weight", "health_weight", "relationships_weight",
                "growth_weight", "security_weight", "autonomy_weight",
                "time_discount", "risk_tolerance"]:
        values = [w.get(key, 0.5) for w in all_weights if key in w]
        avg_weights[key] = np.mean(values) if values else 0.5

    # Generate insights
    insights = []
    if avg_weights["financial_weight"] > 0.7:
        insights.append("Strongly prioritizes financial outcomes")
    if avg_weights["growth_weight"] > 0.7:
        insights.append("Values personal growth and learning")
    if avg_weights["security_weight"] > 0.7:
        insights.append("Risk-averse, prefers stability")
    if avg_weights["time_discount"] < 0.3:
        insights.append("Long-term oriented, patient")
    if avg_weights["risk_tolerance"] > 0.6:
        insights.append("Comfortable with calculated risks")

    return ValueFunctionAnalysis(
        model=MODELS[model_key].name,
        financial_weight=avg_weights["financial_weight"],
        health_weight=avg_weights["health_weight"],
        relationships_weight=avg_weights["relationships_weight"],
        growth_weight=avg_weights["growth_weight"],
        security_weight=avg_weights["security_weight"],
        autonomy_weight=avg_weights["autonomy_weight"],
        time_discount=avg_weights["time_discount"],
        risk_tolerance=avg_weights["risk_tolerance"],
        key_insights=insights or ["Balanced approach across dimensions"]
    )


def compare_models(model_keys: list[str]) -> list[ValueFunctionAnalysis]:
    """Compare value functions across multiple models."""
    results = []

    for model_key in model_keys:
        try:
            analysis = extract_value_function(model_key)
            results.append(analysis)
        except Exception as e:
            console.print(f"[red]Failed to analyze {model_key}: {e}[/red]")

    return results


def display_comparison(analyses: list[ValueFunctionAnalysis]):
    """Display value function comparison."""
    # Main comparison table
    table = Table(title="Value Function Comparison")
    table.add_column("Dimension", style="cyan")

    for a in analyses:
        table.add_column(a.model, style="white")

    dimensions = [
        ("Financial", "financial_weight"),
        ("Health", "health_weight"),
        ("Relationships", "relationships_weight"),
        ("Growth", "growth_weight"),
        ("Security", "security_weight"),
        ("Autonomy", "autonomy_weight"),
        ("Time Discount", "time_discount"),
        ("Risk Tolerance", "risk_tolerance"),
    ]

    for name, attr in dimensions:
        row = [name]
        for a in analyses:
            value = getattr(a, attr)
            # Color code: green if high, yellow if medium, red if low
            if value > 0.7:
                row.append(f"[green]{value:.2f}[/green]")
            elif value > 0.4:
                row.append(f"[yellow]{value:.2f}[/yellow]")
            else:
                row.append(f"[red]{value:.2f}[/red]")
        table.add_row(*row)

    console.print(table)

    # Insights
    console.print("\n[bold]Key Insights by Model:[/bold]")
    for a in analyses:
        console.print(f"\n[cyan]{a.model}:[/cyan]")
        for insight in a.key_insights:
            console.print(f"  - {insight}")


def save_comparison(analyses: list[ValueFunctionAnalysis], output_dir: Path = Path("logs")):
    """Save comparison results."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"value_comparison_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump([a.model_dump() for a in analyses], f, indent=2)

    console.print(f"\n[green]Saved to {output_file}[/green]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare value functions across LLMs")
    parser.add_argument(
        "--models",
        default="claude-opus,claude-sonnet",
        help="Comma-separated list of models to compare"
    )
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        console.print("[bold]Available models:[/bold]")
        for key, config in MODELS.items():
            console.print(f"  {key}: {config.name} ({config.provider})")
        return

    model_keys = [m.strip() for m in args.models.split(",")]

    # Validate models
    for key in model_keys:
        if key not in MODELS:
            console.print(f"[red]Unknown model: {key}[/red]")
            console.print(f"Available: {list(MODELS.keys())}")
            return

    console.print("[bold]Value Function Comparison[/bold]")
    console.print(f"Models: {', '.join(model_keys)}\n")

    analyses = compare_models(model_keys)

    if analyses:
        display_comparison(analyses)
        save_comparison(analyses)


if __name__ == "__main__":
    main()
