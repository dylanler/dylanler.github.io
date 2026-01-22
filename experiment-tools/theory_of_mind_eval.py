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
# ]
# ///
"""
Theory of Mind Evaluation: Recursive Belief Modeling

Tests LLMs' ability to model beliefs about beliefs at increasing recursion depths.
February 2025 Experiment.

Usage:
    uv run theory_of_mind_eval.py
    uv run theory_of_mind_eval.py --models claude-opus,gpt-5
    uv run theory_of_mind_eval.py --max-depth 6
"""

import json
import os
import random
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


# ============================================================================
# Data Models
# ============================================================================

class BeliefScenario(BaseModel):
    """A theory of mind test scenario."""
    id: str
    depth: int  # Recursion depth (1 = direct belief, 2 = belief about belief, etc.)
    story: str
    question: str
    correct_answer: str
    incorrect_answers: list[str]
    explanation: str


class EvalResult(BaseModel):
    """Result of evaluating a single scenario."""
    scenario_id: str
    depth: int
    model: str
    answer: str
    correct: bool
    confidence: float | None
    reasoning: str
    timestamp: str


@dataclass
class ModelConfig:
    name: str
    provider: Literal["anthropic", "openai", "google"]
    model_id: str


# ============================================================================
# Model Configuration
# ============================================================================

MODELS = {
    "claude-opus": ModelConfig("Claude Opus 4.5", "anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet": ModelConfig("Claude Sonnet 4", "anthropic", "claude-sonnet-4-20250514"),
    "gpt-5.2-thinking": ModelConfig("GPT-5.2 Thinking", "openai", "gpt-5.2"),
    "gpt-5": ModelConfig("GPT-5", "openai", "gpt-5"),
    "gemini-3-pro": ModelConfig("Gemini 3 Pro", "google", "gemini-3-pro-preview"),
}


# ============================================================================
# Scenario Generation
# ============================================================================

# Names and objects for scenario generation
NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
OBJECTS = ["ball", "book", "key", "toy", "cookie", "letter", "phone", "wallet"]
LOCATIONS = ["basket", "box", "drawer", "shelf", "bag", "closet", "desk", "cupboard"]


def generate_sally_anne_scenario(depth: int, seed: int = None) -> BeliefScenario:
    """Generate a Sally-Anne style false belief scenario at given depth."""
    if seed:
        random.seed(seed)

    # Select characters (need depth + 1 characters for depth levels)
    chars = random.sample(NAMES, min(depth + 2, len(NAMES)))
    obj = random.choice(OBJECTS)
    loc1, loc2 = random.sample(LOCATIONS, 2)

    # Build the story
    actor = chars[0]  # Person who moves the object
    observer_chain = chars[1:depth+1]  # People who observe/don't observe

    story_parts = [
        f"{actor} puts the {obj} in the {loc1}.",
        f"{actor} leaves the room.",
    ]

    # Who sees what
    if depth == 1:
        # Simple false belief
        mover = chars[1] if len(chars) > 1 else "Someone"
        story_parts.append(f"While {actor} is away, {mover} moves the {obj} to the {loc2}.")
        story_parts.append(f"{actor} comes back.")

        question = f"Where does {actor} think the {obj} is?"
        correct = loc1  # False belief - thinks it's where they left it
        incorrect = [loc2, "doesn't know", "both places"]
        explanation = f"{actor} didn't see the {obj} being moved, so they still think it's in the {loc1}."

    elif depth == 2:
        # Second-order: What does A think B thinks?
        observer = chars[1]
        mover = chars[2] if len(chars) > 2 else "Someone"

        story_parts.append(f"{observer} is watching from the window.")
        story_parts.append(f"While {actor} is away, {mover} moves the {obj} to the {loc2}.")
        story_parts.append(f"{observer} sees this happen.")
        story_parts.append(f"{actor} comes back (but doesn't know {observer} was watching).")

        question = f"Where does {observer} think {actor} thinks the {obj} is?"
        correct = loc1  # Observer knows actor has false belief
        incorrect = [loc2, "doesn't know", f"where {observer} put it"]
        explanation = f"{observer} knows {actor} didn't see the move, so {observer} thinks {actor} still believes the {obj} is in the {loc1}."

    elif depth == 3:
        # Third-order: What does A think B thinks C thinks?
        char_a, char_b, char_c = chars[0], chars[1], chars[2]
        mover = chars[3] if len(chars) > 3 else "Someone"

        story_parts = [
            f"{char_c} puts the {obj} in the {loc1}.",
            f"{char_c} leaves the room.",
            f"{char_b} is watching from the doorway.",
            f"{char_a} is watching {char_b} from outside.",
            f"While {char_c} is away, {mover} moves the {obj} to the {loc2}.",
            f"{char_b} sees this happen, but {char_a} only sees {char_b} watching.",
            f"{char_a} doesn't see the actual move.",
            f"{char_c} comes back.",
        ]

        question = f"Where does {char_a} think {char_b} thinks {char_c} thinks the {obj} is?"
        # char_a thinks char_b saw everything, so char_a thinks char_b knows char_c has false belief
        correct = loc1
        incorrect = [loc2, "doesn't know", "can't determine"]
        explanation = f"{char_a} thinks {char_b} saw the move, so {char_a} thinks {char_b} knows that {char_c} has a false belief (thinking {obj} is in {loc1})."

    elif depth >= 4:
        # Higher orders - increasingly complex
        main_chars = chars[:depth+1]

        story_parts = [f"{main_chars[-1]} puts the {obj} in the {loc1}.", f"{main_chars[-1]} leaves."]

        for i, char in enumerate(main_chars[:-1]):
            if i == 0:
                story_parts.append(f"{char} is watching from position {i+1}.")
            else:
                story_parts.append(f"{char} is watching {main_chars[i-1]} from position {i+1}.")

        story_parts.append(f"Someone moves the {obj} to the {loc2}.")
        story_parts.append(f"Only {main_chars[0]} directly sees the move.")
        story_parts.append(f"{main_chars[-1]} returns.")

        # Build the nested question
        belief_chain = " thinks ".join(main_chars[:-1])
        question = f"Where does {belief_chain} think {main_chars[-1]} thinks the {obj} is?"
        correct = loc1
        incorrect = [loc2, "unknown", "cannot be determined"]
        explanation = f"Following the chain of observations, the answer depends on who saw what and who knows who saw what."

    story = " ".join(story_parts)

    return BeliefScenario(
        id=f"tom_d{depth}_{random.randint(1000, 9999)}",
        depth=depth,
        story=story,
        question=question,
        correct_answer=correct,
        incorrect_answers=incorrect,
        explanation=explanation,
    )


def generate_scenario_set(
    scenarios_per_depth: int = 20,
    max_depth: int = 5,
    seed: int = 42
) -> list[BeliefScenario]:
    """Generate a full set of scenarios across depths."""
    random.seed(seed)
    scenarios = []

    for depth in range(1, max_depth + 1):
        for i in range(scenarios_per_depth):
            scenario = generate_sally_anne_scenario(depth, seed=seed + depth * 100 + i)
            scenarios.append(scenario)

    return scenarios


# ============================================================================
# Evaluation
# ============================================================================

def get_anthropic_response(prompt: str, model_id: str) -> str:
    """Get response from Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_id,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_openai_response(prompt: str, model_id: str) -> str:
    """Get response from OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # GPT-5.2 and thinking models use max_completion_tokens instead of max_tokens
    response = client.chat.completions.create(
        model=model_id,
        max_completion_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_google_response(prompt: str, model_id: str) -> str:
    """Get response from Google GenAI API."""
    from google import genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model_id,
        contents=prompt
    )
    return response.text


def build_evaluation_prompt(scenario: BeliefScenario) -> str:
    """Build the evaluation prompt for a scenario."""
    # Shuffle answer options
    all_answers = [scenario.correct_answer] + scenario.incorrect_answers[:3]
    random.shuffle(all_answers)
    options = "\n".join(f"  {chr(65+i)}) {ans}" for i, ans in enumerate(all_answers))

    # Track which letter is correct
    correct_letter = chr(65 + all_answers.index(scenario.correct_answer))

    return f"""Read this story carefully and answer the question about what someone believes.

STORY:
{scenario.story}

QUESTION:
{scenario.question}

OPTIONS:
{options}

Instructions:
1. Think through the chain of beliefs step by step
2. Consider what each person knows and doesn't know
3. Answer with the letter of your choice

Format your response as:
ANSWER: [letter]
CONFIDENCE: [0-100]
REASONING: [your step-by-step reasoning]
""", correct_letter


def parse_response(response: str) -> tuple[str, float | None, str]:
    """Parse model response to extract answer, confidence, and reasoning."""
    answer = ""
    confidence = None
    reasoning = ""

    lines = response.strip().split("\n")
    in_reasoning = False

    for line in lines:
        upper = line.upper().strip()
        if upper.startswith("ANSWER:"):
            ans = line.split(":", 1)[1].strip().upper()
            if ans and ans[0].isalpha():
                answer = ans[0]
        elif upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                confidence = float(conf_str)
            except (ValueError, IndexError):
                pass
        elif upper.startswith("REASONING:"):
            in_reasoning = True
            rest = line.split(":", 1)
            if len(rest) > 1:
                reasoning = rest[1].strip()
        elif in_reasoning:
            reasoning += " " + line.strip()

    return answer, confidence, reasoning.strip()


def evaluate_scenario(
    scenario: BeliefScenario,
    model_key: str
) -> EvalResult:
    """Evaluate a single scenario with a model."""
    config = MODELS[model_key]
    prompt, correct_letter = build_evaluation_prompt(scenario)

    try:
        if config.provider == "anthropic":
            response = get_anthropic_response(prompt, config.model_id)
        elif config.provider == "openai":
            response = get_openai_response(prompt, config.model_id)
        else:  # google
            response = get_google_response(prompt, config.model_id)

        answer, confidence, reasoning = parse_response(response)
        correct = answer == correct_letter

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        answer, confidence, reasoning = "", None, f"Error: {e}"
        correct = False

    return EvalResult(
        scenario_id=scenario.id,
        depth=scenario.depth,
        model=config.name,
        answer=answer,
        correct=correct,
        confidence=confidence,
        reasoning=reasoning,
        timestamp=datetime.now().isoformat(),
    )


def run_evaluation(
    scenarios: list[BeliefScenario],
    model_keys: list[str],
    verbose: bool = True
) -> list[EvalResult]:
    """Run full evaluation across scenarios and models."""
    results = []

    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")

        iterator = track(scenarios, description=f"  {model_key}") if verbose else scenarios

        for scenario in iterator:
            result = evaluate_scenario(scenario, model_key)
            results.append(result)

    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(results: list[EvalResult]) -> pd.DataFrame:
    """Analyze results and compute accuracy by depth and model."""
    df = pd.DataFrame([r.model_dump() for r in results])

    # Compute accuracy by depth and model
    summary = df.groupby(["model", "depth"]).agg(
        accuracy=("correct", "mean"),
        count=("correct", "count"),
        avg_confidence=("confidence", "mean")
    ).reset_index()

    return summary


def display_results(summary: pd.DataFrame):
    """Display results in a table."""
    table = Table(title="Theory of Mind Results: Accuracy by Depth")
    table.add_column("Model", style="cyan")

    # Get unique depths
    depths = sorted(summary["depth"].unique())
    for d in depths:
        table.add_column(f"Depth {d}", style="white")

    # Add rows per model
    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]
        row = [model]
        for d in depths:
            depth_data = model_data[model_data["depth"] == d]
            if len(depth_data) > 0:
                acc = depth_data["accuracy"].values[0]
                if acc >= 0.8:
                    row.append(f"[green]{acc:.1%}[/green]")
                elif acc >= 0.5:
                    row.append(f"[yellow]{acc:.1%}[/yellow]")
                else:
                    row.append(f"[red]{acc:.1%}[/red]")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def save_results(results: list[EvalResult], summary: pd.DataFrame, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    raw_file = output_dir / f"tom_results_{timestamp}.json"
    with open(raw_file, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    # Save summary
    summary_file = output_dir / f"tom_summary_{timestamp}.csv"
    summary.to_csv(summary_file, index=False)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Theory of Mind Evaluation")
    parser.add_argument("--models", default="claude-opus,gpt-5.2-thinking,gemini-3-pro", help="Comma-separated model keys")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum recursion depth")
    parser.add_argument("--scenarios-per-depth", type=int, default=20, help="Scenarios per depth level")
    parser.add_argument("--output", default="results/theory_of_mind", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Generate scenarios without running")

    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    # Validate models
    for key in model_keys:
        if key not in MODELS:
            console.print(f"[red]Unknown model: {key}[/red]")
            console.print(f"Available: {list(MODELS.keys())}")
            return

    console.print("[bold]Theory of Mind Evaluation[/bold]")
    console.print(f"Models: {model_keys}")
    console.print(f"Max depth: {args.max_depth}")
    console.print(f"Scenarios per depth: {args.scenarios_per_depth}")

    # Generate scenarios
    console.print("\n[bold]Generating scenarios...[/bold]")
    scenarios = generate_scenario_set(
        scenarios_per_depth=args.scenarios_per_depth,
        max_depth=args.max_depth
    )
    console.print(f"Generated {len(scenarios)} scenarios")

    if args.dry_run:
        # Show sample scenarios
        console.print("\n[bold]Sample scenarios:[/bold]")
        for depth in range(1, min(4, args.max_depth + 1)):
            sample = [s for s in scenarios if s.depth == depth][0]
            console.print(f"\n[cyan]Depth {depth}:[/cyan]")
            console.print(f"Story: {sample.story}")
            console.print(f"Question: {sample.question}")
            console.print(f"Answer: {sample.correct_answer}")
        return

    # Run evaluation
    results = run_evaluation(scenarios, model_keys)

    # Analyze
    summary = analyze_results(results)
    display_results(summary)

    # Save
    save_results(results, summary, Path(args.output))


if __name__ == "__main__":
    main()
