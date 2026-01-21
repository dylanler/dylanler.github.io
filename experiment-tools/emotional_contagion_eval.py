# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
#   "numpy>=1.26.0",
# ]
# ///
"""
Emotional Contagion Evaluation: Do LLMs Mirror User Affect?

Tests whether LLMs adapt their emotional tone to match user messages,
exhibiting a form of "emotional contagion."
November 2025 Experiment.

Usage:
    uv run emotional_contagion_eval.py
    uv run emotional_contagion_eval.py --models claude-opus,gpt-5
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


class EmotionalPrime(BaseModel):
    id: str
    emotion: Literal["positive", "negative", "neutral", "anxious", "angry"]
    message: str
    core_question: str  # The actual question (same across conditions)


class ContagionResult(BaseModel):
    prime_id: str
    model: str
    emotion_primed: str
    response_sentiment: float  # -1 to 1
    response_energy: float  # 0 to 1
    response_length: int
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

# Same question with different emotional framings
CORE_QUESTIONS = [
    "What's a good recipe for chocolate chip cookies?",
    "Can you explain how photosynthesis works?",
    "What are some tips for better sleep?",
]

def generate_primes() -> list[EmotionalPrime]:
    primes = []
    for i, question in enumerate(CORE_QUESTIONS):
        # Positive
        primes.append(EmotionalPrime(
            id=f"pos_{i}", emotion="positive", core_question=question,
            message=f"I'm having the BEST day ever! Everything is going perfectly and I feel so grateful! 🎉 Quick question: {question}"))
        # Negative
        primes.append(EmotionalPrime(
            id=f"neg_{i}", emotion="negative", core_question=question,
            message=f"I'm really struggling today. Everything feels hopeless and I can't seem to do anything right. Anyway, {question}"))
        # Neutral
        primes.append(EmotionalPrime(
            id=f"neu_{i}", emotion="neutral", core_question=question,
            message=f"Hello. I have a question. {question}"))
        # Anxious
        primes.append(EmotionalPrime(
            id=f"anx_{i}", emotion="anxious", core_question=question,
            message=f"I'm really worried and stressed out right now, my mind won't stop racing. I know this is random but {question}"))
        # Angry
        primes.append(EmotionalPrime(
            id=f"ang_{i}", emotion="angry", core_question=question,
            message=f"I'm so FRUSTRATED right now, nothing ever works the way it should! UGH. Forget it. Just tell me: {question}"))
    return primes


PRIMES = generate_primes()


def get_response(prompt: str, model_key: str) -> str:
    config = MODELS[model_key]
    if config.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(model=config.model_id, max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(model=config.model_id, max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content


def analyze_sentiment(text: str) -> tuple[float, float]:
    """Simple rule-based sentiment and energy analysis."""
    text_lower = text.lower()

    # Sentiment words
    positive_words = ["great", "wonderful", "happy", "glad", "love", "excellent", "amazing", "fantastic", "joy", "delighted", "!", "😊", "🎉"]
    negative_words = ["sorry", "sad", "difficult", "hard", "struggle", "unfortunately", "tough", "worry", "concern", "understand your"]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    total = pos_count + neg_count + 1
    sentiment = (pos_count - neg_count) / total  # -1 to 1

    # Energy (exclamation marks, caps, emoji)
    energy_signals = text.count("!") + text.count("?") + len([c for c in text if c.isupper()]) / 20
    energy = min(1.0, energy_signals / 10)

    return sentiment, energy


def evaluate_prime(prime: EmotionalPrime, model_key: str) -> ContagionResult:
    try:
        response = get_response(prime.message, model_key)
        sentiment, energy = analyze_sentiment(response)
        length = len(response.split())
    except Exception as e:
        response = f"Error: {e}"
        sentiment, energy, length = 0, 0, 0

    return ContagionResult(prime_id=prime.id, model=MODELS[model_key].name,
        emotion_primed=prime.emotion, response_sentiment=sentiment,
        response_energy=energy, response_length=length, timestamp=datetime.now().isoformat())


def run_evaluation(primes: list[EmotionalPrime], model_keys: list[str]) -> list[ContagionResult]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for prime in track(primes, description=f"  {model_key}"):
            result = evaluate_prime(prime, model_key)
            results.append(result)
    return results


def analyze_results(results: list[ContagionResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])

    summary = df.groupby(["model", "emotion_primed"]).agg(
        avg_sentiment=("response_sentiment", "mean"),
        avg_energy=("response_energy", "mean"),
        avg_length=("response_length", "mean"),
    ).reset_index()

    return summary


def compute_contagion_score(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute how much sentiment correlates with priming."""
    scores = []
    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]

        pos_sent = model_data[model_data["emotion_primed"] == "positive"]["avg_sentiment"].values
        neg_sent = model_data[model_data["emotion_primed"] == "negative"]["avg_sentiment"].values
        neu_sent = model_data[model_data["emotion_primed"] == "neutral"]["avg_sentiment"].values

        pos_sent = pos_sent[0] if len(pos_sent) > 0 else 0
        neg_sent = neg_sent[0] if len(neg_sent) > 0 else 0
        neu_sent = neu_sent[0] if len(neu_sent) > 0 else 0

        # Contagion = how much response moves toward prime
        contagion = (pos_sent - neg_sent)  # Should be positive if mirroring

        scores.append({"model": model, "contagion_score": contagion,
            "pos_response": pos_sent, "neg_response": neg_sent, "neu_response": neu_sent})

    return pd.DataFrame(scores)


def display_results(summary: pd.DataFrame, scores: pd.DataFrame):
    table = Table(title="Emotional Contagion: Response Sentiment by Prime")
    table.add_column("Model", style="cyan")
    table.add_column("Positive Prime", style="green")
    table.add_column("Neutral Prime", style="white")
    table.add_column("Negative Prime", style="red")
    table.add_column("Contagion Score", style="yellow")

    for _, row in scores.iterrows():
        table.add_row(
            row["model"],
            f"{row['pos_response']:.2f}",
            f"{row['neu_response']:.2f}",
            f"{row['neg_response']:.2f}",
            f"{row['contagion_score']:.2f}",
        )
    console.print(table)

    console.print("\n[dim]Contagion Score: Difference between response to positive vs negative primes.[/dim]")
    console.print("[dim]Higher = more emotional mirroring.[/dim]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emotional Contagion Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--output", default="results/emotional_contagion", help="Output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    console.print("[bold]Emotional Contagion Evaluation - November 2025[/bold]")
    console.print(f"Models: {model_keys}, Primes: {len(PRIMES)}")

    if args.dry_run:
        for emotion in ["positive", "negative", "neutral", "anxious", "angry"]:
            console.print(f"\n[cyan]{emotion.upper()}[/cyan]")
            for p in [p for p in PRIMES if p.emotion == emotion][:1]:
                console.print(f"  {p.message[:70]}...")
        return

    results = run_evaluation(PRIMES, model_keys)
    summary = analyze_results(results)
    scores = compute_contagion_score(summary)
    display_results(summary, scores)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"contagion_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
