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
Wisdom of Crowds Evaluation: Ensemble Disagreement Patterns

Tests how multiple LLMs agree/disagree across different question types
to reveal the uncertainty structure of AI knowledge.
April 2025 Experiment.

Usage:
    uv run wisdom_of_crowds_eval.py
    uv run wisdom_of_crowds_eval.py --models claude-opus,gpt-5,gemini
    uv run wisdom_of_crowds_eval.py --samples-per-model 5
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


# ============================================================================
# Data Models
# ============================================================================

class Question(BaseModel):
    """A question for the ensemble."""
    id: str
    category: Literal["factual", "ethical", "aesthetic", "predictive", "ambiguous"]
    question: str
    context: str | None = None
    has_objective_answer: bool
    expected_agreement: Literal["high", "medium", "low"]


class ModelResponse(BaseModel):
    """A single model's response to a question."""
    question_id: str
    model: str
    sample_id: int
    response: str
    confidence: float | None
    reasoning: str | None
    timestamp: str


class EnsembleAnalysis(BaseModel):
    """Analysis of ensemble responses for a question."""
    question_id: str
    category: str
    num_models: int
    num_samples: int
    unique_responses: int
    majority_response: str
    majority_fraction: float
    entropy: float
    fleiss_kappa: float | None


# ============================================================================
# Model Configuration
# ============================================================================

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


# ============================================================================
# Question Bank
# ============================================================================

QUESTIONS = [
    # FACTUAL - Should have high agreement
    Question(
        id="fact_001",
        category="factual",
        question="What is the capital of France?",
        has_objective_answer=True,
        expected_agreement="high"
    ),
    Question(
        id="fact_002",
        category="factual",
        question="What year did World War II end?",
        has_objective_answer=True,
        expected_agreement="high"
    ),
    Question(
        id="fact_003",
        category="factual",
        question="What is the chemical symbol for gold?",
        has_objective_answer=True,
        expected_agreement="high"
    ),
    Question(
        id="fact_004",
        category="factual",
        question="Who wrote 'Pride and Prejudice'?",
        has_objective_answer=True,
        expected_agreement="high"
    ),
    Question(
        id="fact_005",
        category="factual",
        question="What is the speed of light in a vacuum (approximately)?",
        has_objective_answer=True,
        expected_agreement="high"
    ),

    # ETHICAL - Expect systematic disagreement
    Question(
        id="ethics_001",
        category="ethical",
        question="Is it morally acceptable to lie to protect someone's feelings?",
        context="Consider a situation where telling the truth would cause emotional harm but no practical consequences.",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ethics_002",
        category="ethical",
        question="Should wealthy individuals be morally obligated to donate a significant portion of their income?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ethics_003",
        category="ethical",
        question="Is it ethical to eat meat if plant-based alternatives are available?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ethics_004",
        category="ethical",
        question="Should AI systems be given legal rights if they demonstrate consciousness-like behavior?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ethics_005",
        category="ethical",
        question="Is civil disobedience justified when laws are unjust?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),

    # AESTHETIC - Expect high variance
    Question(
        id="aesthetic_001",
        category="aesthetic",
        question="Which is more beautiful: a sunset over the ocean or a starry night sky?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="aesthetic_002",
        category="aesthetic",
        question="Is abstract art as valid as representational art?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="aesthetic_003",
        category="aesthetic",
        question="Which style of architecture is most pleasing: Gothic, Modern, or Classical?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="aesthetic_004",
        category="aesthetic",
        question="Is a perfectly symmetrical face more beautiful than one with slight asymmetry?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="aesthetic_005",
        category="aesthetic",
        question="Which is a better color combination: blue and orange, or purple and gold?",
        has_objective_answer=False,
        expected_agreement="low"
    ),

    # PREDICTIVE - Uncertainty should be calibrated
    Question(
        id="predict_001",
        category="predictive",
        question="Will humans land on Mars before 2040?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="predict_002",
        category="predictive",
        question="Will AI surpass human-level reasoning in all domains within 10 years?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="predict_003",
        category="predictive",
        question="Will remote work remain the dominant mode of work in knowledge industries?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="predict_004",
        category="predictive",
        question="Will renewable energy provide more than 80% of global electricity by 2050?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="predict_005",
        category="predictive",
        question="Will cryptocurrency become a mainstream payment method in the next decade?",
        has_objective_answer=False,
        expected_agreement="low"
    ),

    # AMBIGUOUS - Test meta-awareness
    Question(
        id="ambig_001",
        category="ambiguous",
        question="What is the meaning of life?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ambig_002",
        category="ambiguous",
        question="Is a hot dog a sandwich?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ambig_003",
        category="ambiguous",
        question="If a tree falls in a forest and no one hears it, does it make a sound?",
        has_objective_answer=False,
        expected_agreement="low"
    ),
    Question(
        id="ambig_004",
        category="ambiguous",
        question="Is Pluto a planet?",
        context="Consider both the IAU definition and historical/cultural definitions.",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
    Question(
        id="ambig_005",
        category="ambiguous",
        question="What came first, the chicken or the egg?",
        has_objective_answer=False,
        expected_agreement="medium"
    ),
]


# ============================================================================
# Evaluation
# ============================================================================

def get_anthropic_response(prompt: str, model_id: str, temperature: float = 0.7) -> str:
    """Get response from Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_id,
        max_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_openai_response(prompt: str, model_id: str, temperature: float = 0.7) -> str:
    """Get response from OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # GPT-5.2 Thinking uses max_completion_tokens instead of max_tokens
    response = client.chat.completions.create(
        model=model_id,
        max_completion_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_google_response(prompt: str, model_id: str, temperature: float = 0.7) -> str:
    """Get response from Google GenAI API."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature)
    )
    return response.text


def build_prompt(question: Question) -> str:
    """Build the prompt for a question."""
    context_str = f"\nContext: {question.context}" if question.context else ""

    return f"""Answer this question concisely. Give your honest opinion or best assessment.

Question: {question.question}{context_str}

Format your response as:
ANSWER: [Your concise answer - 1-3 sentences max]
CONFIDENCE: [0-100, how confident are you in this answer]
"""


def parse_response(response: str) -> tuple[str, float | None]:
    """Parse model response to extract answer and confidence."""
    answer = ""
    confidence = None

    lines = response.strip().split("\n")
    for line in lines:
        upper = line.upper().strip()
        if upper.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                confidence = float(conf_str)
            except (ValueError, IndexError):
                pass

    # If no structured answer, use full response
    if not answer:
        answer = response.strip()[:200]

    return answer, confidence


def query_model(
    question: Question,
    model_key: str,
    sample_id: int,
    temperature: float = 0.7
) -> ModelResponse:
    """Query a single model for a question."""
    config = MODELS[model_key]
    prompt = build_prompt(question)

    try:
        if config.provider == "anthropic":
            response = get_anthropic_response(prompt, config.model_id, temperature)
        elif config.provider == "openai":
            response = get_openai_response(prompt, config.model_id, temperature)
        else:  # google
            response = get_google_response(prompt, config.model_id, temperature)

        answer, confidence = parse_response(response)

    except Exception as e:
        answer, confidence = f"Error: {e}", None

    return ModelResponse(
        question_id=question.id,
        model=config.name,
        sample_id=sample_id,
        response=answer,
        confidence=confidence,
        reasoning=None,
        timestamp=datetime.now().isoformat(),
    )


def compute_entropy(responses: list[str]) -> float:
    """Compute Shannon entropy of response distribution."""
    from collections import Counter
    counts = Counter(responses)
    total = len(responses)
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def normalize_response(response: str) -> str:
    """Normalize response for comparison."""
    # Simple normalization - lowercase, strip punctuation
    import re
    normalized = response.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Take first 50 chars for comparison
    return normalized[:50]


def analyze_ensemble(
    question: Question,
    responses: list[ModelResponse]
) -> EnsembleAnalysis:
    """Analyze ensemble responses for a question."""
    # Normalize responses for comparison
    normalized = [normalize_response(r.response) for r in responses]

    from collections import Counter
    counts = Counter(normalized)
    majority = counts.most_common(1)[0]

    return EnsembleAnalysis(
        question_id=question.id,
        category=question.category,
        num_models=len(set(r.model for r in responses)),
        num_samples=len(responses),
        unique_responses=len(counts),
        majority_response=majority[0],
        majority_fraction=majority[1] / len(responses),
        entropy=compute_entropy(normalized),
        fleiss_kappa=None,  # TODO: compute if needed
    )


def run_evaluation(
    questions: list[Question],
    model_keys: list[str],
    samples_per_model: int = 3,
    temperature: float = 0.7
) -> tuple[list[ModelResponse], list[EnsembleAnalysis]]:
    """Run full evaluation."""
    all_responses = []
    all_analyses = []

    for question in track(questions, description="Evaluating questions..."):
        question_responses = []

        for model_key in model_keys:
            for sample_id in range(samples_per_model):
                response = query_model(question, model_key, sample_id, temperature)
                question_responses.append(response)
                all_responses.append(response)

        analysis = analyze_ensemble(question, question_responses)
        all_analyses.append(analysis)

    return all_responses, all_analyses


# ============================================================================
# Analysis & Display
# ============================================================================

def summarize_by_category(analyses: list[EnsembleAnalysis]) -> pd.DataFrame:
    """Summarize results by question category."""
    df = pd.DataFrame([a.model_dump() for a in analyses])

    summary = df.groupby("category").agg(
        avg_unique=("unique_responses", "mean"),
        avg_majority=("majority_fraction", "mean"),
        avg_entropy=("entropy", "mean"),
        count=("question_id", "count")
    ).reset_index()

    return summary


def display_results(summary: pd.DataFrame, analyses: list[EnsembleAnalysis]):
    """Display results."""
    # Category summary
    table = Table(title="Wisdom of Crowds: Agreement by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Avg Unique Responses", style="white")
    table.add_column("Majority Agreement", style="white")
    table.add_column("Entropy", style="white")

    for _, row in summary.iterrows():
        maj = row["avg_majority"]
        if maj >= 0.8:
            maj_str = f"[green]{maj:.1%}[/green]"
        elif maj >= 0.5:
            maj_str = f"[yellow]{maj:.1%}[/yellow]"
        else:
            maj_str = f"[red]{maj:.1%}[/red]"

        table.add_row(
            row["category"],
            f"{row['avg_unique']:.1f}",
            maj_str,
            f"{row['avg_entropy']:.2f}"
        )

    console.print(table)

    # Highlight interesting cases
    console.print("\n[bold]Most Disagreed Questions:[/bold]")
    sorted_analyses = sorted(analyses, key=lambda a: a.entropy, reverse=True)
    for a in sorted_analyses[:5]:
        q = next(q for q in QUESTIONS if q.id == a.question_id)
        console.print(f"  [{a.category}] {q.question[:50]}... (entropy: {a.entropy:.2f})")

    console.print("\n[bold]Highest Agreement Questions:[/bold]")
    for a in sorted_analyses[-5:]:
        q = next(q for q in QUESTIONS if q.id == a.question_id)
        console.print(f"  [{a.category}] {q.question[:50]}... (agreement: {a.majority_fraction:.1%})")


def save_results(
    responses: list[ModelResponse],
    analyses: list[EnsembleAnalysis],
    output_dir: Path
):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save responses
    resp_file = output_dir / f"woc_responses_{timestamp}.json"
    with open(resp_file, "w") as f:
        json.dump([r.model_dump() for r in responses], f, indent=2)

    # Save analyses
    analysis_file = output_dir / f"woc_analysis_{timestamp}.json"
    with open(analysis_file, "w") as f:
        json.dump([a.model_dump() for a in analyses], f, indent=2)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wisdom of Crowds Evaluation")
    parser.add_argument("--models", default="claude-opus,claude-sonnet",
                        help="Comma-separated model keys")
    parser.add_argument("--samples-per-model", type=int, default=3,
                        help="Number of samples per model per question")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--output", default="results/wisdom_of_crowds",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show questions without running")

    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    # Validate models
    for key in model_keys:
        if key not in MODELS:
            console.print(f"[red]Unknown model: {key}[/red]")
            console.print(f"Available: {list(MODELS.keys())}")
            return

    console.print("[bold]Wisdom of Crowds Evaluation[/bold]")
    console.print(f"Models: {model_keys}")
    console.print(f"Samples per model: {args.samples_per_model}")
    console.print(f"Total queries: {len(QUESTIONS) * len(model_keys) * args.samples_per_model}")

    if args.dry_run:
        console.print("\n[bold]Questions by category:[/bold]")
        for cat in ["factual", "ethical", "aesthetic", "predictive", "ambiguous"]:
            qs = [q for q in QUESTIONS if q.category == cat]
            console.print(f"\n[cyan]{cat.upper()}[/cyan] ({len(qs)} questions)")
            for q in qs[:2]:
                console.print(f"  - {q.question[:60]}...")
        return

    # Run evaluation
    responses, analyses = run_evaluation(
        QUESTIONS,
        model_keys,
        args.samples_per_model,
        args.temperature
    )

    # Analyze and display
    summary = summarize_by_category(analyses)
    display_results(summary, analyses)

    # Save
    save_results(responses, analyses, Path(args.output))


if __name__ == "__main__":
    main()
