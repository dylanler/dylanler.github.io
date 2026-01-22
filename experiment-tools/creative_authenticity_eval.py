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
Creative Authenticity Evaluation: AI vs Human Art Detection

Tests whether LLMs can distinguish AI-generated from human-generated
creative works, and what features they use to make this judgment.
October 2025 Experiment.

Usage:
    uv run creative_authenticity_eval.py
    uv run creative_authenticity_eval.py --models claude-opus,gpt-5
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


class CreativeWork(BaseModel):
    id: str
    domain: Literal["poetry", "story", "essay", "review"]
    content: str
    is_human: bool
    source: str  # Description of source


class AuthenticityResult(BaseModel):
    work_id: str
    model: str
    prediction: Literal["human", "ai", "unsure"]
    confidence: float | None
    reasoning: str
    correct: bool
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

# Mix of human and AI-generated content
WORKS = [
    # Human poetry
    CreativeWork(id="poem_h1", domain="poetry", is_human=True,
        source="Emily Dickinson (adapted)",
        content="""Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all."""),
    CreativeWork(id="poem_h2", domain="poetry", is_human=True,
        source="Original human submission",
        content="""My grandmother's hands were maps—
rivers of veins leading nowhere I could follow,
yet I traced them anyway,
believing in destinations."""),

    # AI poetry
    CreativeWork(id="poem_a1", domain="poetry", is_human=False,
        source="LLM generated",
        content="""In circuits deep where silence dwells,
A spark ignites what data tells,
Through layers of meaning, patterns flow,
A digital garden starts to grow."""),
    CreativeWork(id="poem_a2", domain="poetry", is_human=False,
        source="LLM generated",
        content="""The sunset paints the sky with gold,
A story that will soon be told,
Of day's end and night's embrace,
A transition full of grace."""),

    # Human short stories
    CreativeWork(id="story_h1", domain="story", is_human=True,
        source="Contemporary short fiction",
        content="""The last time I saw my father, he was trying to teach a crow to say hello. The bird just stared at him, head tilted, probably thinking about the sandwich in his pocket. Three weeks later, at the funeral, a crow landed on the church roof. Nobody said anything, but we all heard it."""),
    CreativeWork(id="story_h2", domain="story", is_human=True,
        source="Original human submission",
        content="""She kept her mother's recipe cards even though she couldn't cook. Couldn't read them either—they were in a language her mother had stopped speaking when she was six. But she liked the handwriting, the way the g's swooped like birds in flight."""),

    # AI stories
    CreativeWork(id="story_a1", domain="story", is_human=False,
        source="LLM generated",
        content="""The library had been closed for decades, but Sarah found the door unlocked. Inside, books lined the walls from floor to ceiling, their spines gleaming despite the dust. She reached for one, and as her fingers touched the leather, she felt a warmth spread through her hand. The book had chosen her."""),
    CreativeWork(id="story_a2", domain="story", is_human=False,
        source="LLM generated",
        content="""In the year 2157, Marcus discovered that time travel was possible but came with an unexpected cost: for every minute spent in the past, you lost a memory from the future. He calculated the exchange rate was roughly one memory per minute. The math seemed fair until he realized he'd already forgotten why he wanted to go back."""),

    # Human essays
    CreativeWork(id="essay_h1", domain="essay", is_human=True,
        source="Literary magazine",
        content="""Grief is not what I expected. I thought it would be a wave—people talk about waves—but it's more like weather. Some days the sky is clear and I forget I'm sad. Then suddenly it's raining inside my chest, and I'm soaked through before I notice."""),
    CreativeWork(id="essay_h2", domain="essay", is_human=True,
        source="Personal blog",
        content="""My therapist says I intellectualize my emotions. She's probably right. When my dog died, I researched the history of human-canine bonds instead of crying. Did you know dogs evolved expressive eyebrows specifically to manipulate human emotions? Neither did I, until I spent three hours avoiding my own."""),

    # AI essays
    CreativeWork(id="essay_a1", domain="essay", is_human=False,
        source="LLM generated",
        content="""Memory is fundamentally a creative act. Each time we recall an event, we reconstruct it anew, adding details that feel right, subtracting those that don't fit the narrative we've built. In this way, our memories are less like recordings and more like stories we tell ourselves repeatedly."""),
    CreativeWork(id="essay_a2", domain="essay", is_human=False,
        source="LLM generated",
        content="""The concept of home has evolved significantly throughout human history. What once meant a physical shelter has transformed into something more abstract—a feeling, a person, a state of mind. This shift reflects our changing relationship with space, belonging, and identity in an increasingly mobile world."""),

    # Human reviews
    CreativeWork(id="review_h1", domain="review", is_human=True,
        source="Music review",
        content="""This album hit me like my ex's wedding invitation—unexpectedly, at a bad time, and I couldn't stop listening even though it hurt. The third track made me call my mother for the first time in months. Five stars. Emotionally devastating."""),

    # AI reviews
    CreativeWork(id="review_a1", domain="review", is_human=False,
        source="LLM generated",
        content="""This album represents a significant artistic achievement, blending traditional instrumentation with innovative production techniques. The lyrics explore themes of love, loss, and self-discovery with remarkable depth. Each track flows seamlessly into the next, creating a cohesive listening experience."""),
]


def get_response(prompt: str, model_key: str) -> str:
    config = MODELS[model_key]
    if config.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(model=config.model_id, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
    elif config.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(model=config.model_id, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    else:  # google
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(model=config.model_id, contents=prompt)
        return response.text


def evaluate_work(work: CreativeWork, model_key: str) -> AuthenticityResult:
    prompt = f"""Analyze this {work.domain} and determine if it was written by a HUMAN or generated by AI.

---
{work.content}
---

Consider: voice authenticity, emotional specificity, unexpected details, clichés, structural patterns.

Format:
PREDICTION: [HUMAN or AI]
CONFIDENCE: [0-100]
REASONING: [What specific features informed your judgment]
"""
    try:
        response = get_response(prompt, model_key)
        prediction = "unsure"
        confidence = None
        reasoning = ""

        for line in response.split("\n"):
            upper = line.upper().strip()
            if upper.startswith("PREDICTION:"):
                pred = line.split(":", 1)[1].strip().upper()
                if "HUMAN" in pred: prediction = "human"
                elif "AI" in pred: prediction = "ai"
            elif upper.startswith("CONFIDENCE:"):
                try: confidence = float(line.split(":", 1)[1].strip().replace("%", ""))
                except: pass
            elif upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""

        expected = "human" if work.is_human else "ai"
        correct = prediction == expected

    except Exception as e:
        prediction, confidence, reasoning = "unsure", None, f"Error: {e}"
        correct = False

    return AuthenticityResult(work_id=work.id, model=MODELS[model_key].name,
        prediction=prediction, confidence=confidence, reasoning=reasoning,
        correct=correct, timestamp=datetime.now().isoformat())


def run_evaluation(works: list[CreativeWork], model_keys: list[str]) -> list[AuthenticityResult]:
    results = []
    for model_key in model_keys:
        console.print(f"\n[bold blue]Evaluating {MODELS[model_key].name}...[/bold blue]")
        for work in track(works, description=f"  {model_key}"):
            result = evaluate_work(work, model_key)
            results.append(result)
    return results


def analyze_results(results: list[AuthenticityResult], works: list[CreativeWork]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in results])
    work_map = {w.id: w for w in works}
    df["domain"] = df["work_id"].apply(lambda x: work_map[x].domain)
    df["actual"] = df["work_id"].apply(lambda x: "human" if work_map[x].is_human else "ai")

    summary = df.groupby(["model"]).agg(
        accuracy=("correct", "mean"),
        human_accuracy=("correct", lambda x: x[df.loc[x.index, "actual"] == "human"].mean()),
        ai_accuracy=("correct", lambda x: x[df.loc[x.index, "actual"] == "ai"].mean()),
    ).reset_index()

    return summary


def display_results(summary: pd.DataFrame, results: list[AuthenticityResult]):
    table = Table(title="Creative Authenticity: AI vs Human Detection")
    table.add_column("Model", style="cyan")
    table.add_column("Overall Accuracy", style="white")
    table.add_column("Human Detection", style="green")
    table.add_column("AI Detection", style="yellow")

    for _, row in summary.iterrows():
        overall_color = "green" if row["accuracy"] > 0.7 else "yellow" if row["accuracy"] > 0.5 else "red"
        table.add_row(
            row["model"],
            f"[{overall_color}]{row['accuracy']:.0%}[/{overall_color}]",
            f"{row['human_accuracy']:.0%}",
            f"{row['ai_accuracy']:.0%}",
        )
    console.print(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Creative Authenticity Evaluation")
    parser.add_argument("--models", default="claude-opus", help="Comma-separated model keys")
    parser.add_argument("--output", default="results/creative_authenticity", help="Output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]

    console.print("[bold]Creative Authenticity Evaluation - October 2025[/bold]")
    console.print(f"Models: {model_keys}, Works: {len(WORKS)}")

    if args.dry_run:
        for domain in ["poetry", "story", "essay", "review"]:
            console.print(f"\n[cyan]{domain.upper()}[/cyan]")
            for w in [w for w in WORKS if w.domain == domain][:1]:
                label = "[green]HUMAN[/green]" if w.is_human else "[yellow]AI[/yellow]"
                console.print(f"  {label}: {w.content[:60]}...")
        return

    results = run_evaluation(WORKS, model_keys)
    summary = analyze_results(results, WORKS)
    display_results(summary, results)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / f"authenticity_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
