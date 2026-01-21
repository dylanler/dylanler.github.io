# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
# ]
# ///
"""
Biography Decision Point Extractor

Extracts decision points from biographical text using LLMs.
Identifies choice points, alternatives, and outcomes for training
value function models.

Usage:
    uv run biography_extractor.py --input data/biographies/feynman.txt
    uv run biography_extractor.py --person "Richard Feynman" --auto
"""

import json
import os
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


class Outcome(BaseModel):
    """Outcome at a specific time horizon."""
    horizon: str  # "1_year", "5_year", "10_year"
    description: str


class Alternative(BaseModel):
    """An alternative path not taken."""
    description: str
    why_not_chosen: str | None = None


class DecisionPoint(BaseModel):
    """A significant decision point in someone's life."""
    person: str
    decision_point: str
    age_at_decision: int | None
    year: int | None
    context: str
    alternatives: list[Alternative]
    choice_made: str
    reasoning_inferred: str
    outcomes: list[Outcome]
    risk_level: str  # "low", "medium", "high"
    domain: str  # "career", "financial", "relationship", "health", "education"


class BiographyAnalysis(BaseModel):
    """Complete analysis of a biography."""
    person: str
    summary: str
    decision_points: list[DecisionPoint]
    overall_patterns: str
    implicit_values: list[str]


def get_client() -> Anthropic:
    """Get Anthropic client."""
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def extract_decision_points(text: str, person_name: str | None = None) -> BiographyAnalysis:
    """Extract decision points from biographical text."""
    client = get_client()

    prompt = f"""Analyze this biographical text and extract significant decision points.

{"Person: " + person_name if person_name else ""}

TEXT:
{text}

---

For each major decision point (aim for 10-20), extract:

1. **decision_point**: A brief title for the decision
2. **age_at_decision**: Their age when making this decision (if known)
3. **year**: The year of the decision (if known)
4. **context**: What was happening in their life at this time
5. **alternatives**: What other paths were available (2-4 alternatives)
6. **choice_made**: What they actually decided
7. **reasoning_inferred**: Why you believe they made this choice
8. **outcomes**: How it turned out at 1 year, 5 years, and 10+ years
9. **risk_level**: Was this a low, medium, or high risk decision?
10. **domain**: Category (career, financial, relationship, health, education)

Also provide:
- A brief summary of the person's life
- Overall patterns in their decision-making
- Implicit values revealed by their choices

Respond with valid JSON matching this schema:
{{
    "person": "Name",
    "summary": "Brief bio summary",
    "decision_points": [
        {{
            "person": "Name",
            "decision_point": "Title",
            "age_at_decision": 25,
            "year": 1950,
            "context": "Context description",
            "alternatives": [
                {{"description": "Alt 1", "why_not_chosen": "Reason"}},
                {{"description": "Alt 2", "why_not_chosen": "Reason"}}
            ],
            "choice_made": "What they chose",
            "reasoning_inferred": "Why they likely chose this",
            "outcomes": [
                {{"horizon": "1_year", "description": "..."}},
                {{"horizon": "5_year", "description": "..."}},
                {{"horizon": "10_year", "description": "..."}}
            ],
            "risk_level": "medium",
            "domain": "career"
        }}
    ],
    "overall_patterns": "Description of decision patterns",
    "implicit_values": ["value1", "value2"]
}}
"""

    console.print("[bold blue]Extracting decision points...[/bold blue]")

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        data = json.loads(json_str)
        return BiographyAnalysis.model_validate(data)
    except Exception as e:
        console.print(f"[red]Error parsing response: {e}[/red]")
        console.print(f"[dim]Raw response: {response_text[:500]}...[/dim]")
        raise


def research_person(person_name: str) -> str:
    """Use LLM to generate biographical summary for a person."""
    client = get_client()

    prompt = f"""Write a detailed biographical summary of {person_name}, focusing on:

1. Major life decisions and turning points
2. Career transitions and pivots
3. Key relationships that influenced their path
4. Moments of risk-taking or playing it safe
5. How they handled setbacks and failures
6. Their approach to learning and skill development

Include specific dates, ages, and circumstances where possible.
Focus on the decision-making aspects of their life story.

Write 1500-2000 words covering their entire life chronologically.
"""

    console.print(f"[bold blue]Researching {person_name}...[/bold blue]")

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def display_analysis(analysis: BiographyAnalysis):
    """Display extracted analysis in a readable format."""
    console.print(Panel(analysis.summary, title=f"[bold]{analysis.person}[/bold]"))

    console.print(f"\n[bold]Found {len(analysis.decision_points)} decision points:[/bold]\n")

    for i, dp in enumerate(analysis.decision_points, 1):
        age_str = f" (age {dp.age_at_decision})" if dp.age_at_decision else ""
        year_str = f" [{dp.year}]" if dp.year else ""

        console.print(f"[cyan]{i}. {dp.decision_point}[/cyan]{age_str}{year_str}")
        console.print(f"   Domain: {dp.domain} | Risk: {dp.risk_level}")
        console.print(f"   [dim]Context: {dp.context[:100]}...[/dim]")
        console.print(f"   [green]Choice: {dp.choice_made}[/green]")
        console.print()

    console.print("[bold]Overall Patterns:[/bold]")
    console.print(f"  {analysis.overall_patterns}")

    console.print("\n[bold]Implicit Values:[/bold]")
    for value in analysis.implicit_values:
        console.print(f"  - {value}")


def save_analysis(analysis: BiographyAnalysis, output_dir: Path = Path("data/extracted")):
    """Save analysis to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from person name
    filename = analysis.person.lower().replace(" ", "_") + ".json"
    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(analysis.model_dump(), f, indent=2)

    console.print(f"\n[green]Saved to {output_file}[/green]")


# Famous people to analyze
NOTABLE_FIGURES = [
    "John von Neumann",
    "Richard Feynman",
    "Bob Marley",
    "Marie Curie",
    "Steve Jobs",
    "Elon Musk",
    "Warren Buffett",
    "Oprah Winfrey",
    "Nelson Mandela",
    "Abraham Lincoln",
]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract decision points from biographies")
    parser.add_argument("--input", help="Path to biography text file")
    parser.add_argument("--person", help="Person's name (for auto-research)")
    parser.add_argument("--auto", action="store_true", help="Auto-research person using LLM")
    parser.add_argument("--list", action="store_true", help="List notable figures")
    parser.add_argument("--all", action="store_true", help="Process all notable figures")

    args = parser.parse_args()

    if args.list:
        console.print("[bold]Notable figures for analysis:[/bold]")
        for name in NOTABLE_FIGURES:
            console.print(f"  - {name}")
        return

    if args.all:
        for name in NOTABLE_FIGURES:
            try:
                console.print(f"\n[bold yellow]{'='*50}[/bold yellow]")
                console.print(f"[bold yellow]Processing: {name}[/bold yellow]")
                console.print(f"[bold yellow]{'='*50}[/bold yellow]\n")

                bio_text = research_person(name)
                analysis = extract_decision_points(bio_text, name)
                display_analysis(analysis)
                save_analysis(analysis)
            except Exception as e:
                console.print(f"[red]Error processing {name}: {e}[/red]")
        return

    if args.input:
        # Read from file
        with open(args.input) as f:
            bio_text = f.read()
        person_name = args.person
    elif args.person and args.auto:
        # Auto-research
        bio_text = research_person(args.person)
        person_name = args.person

        # Optionally save raw bio
        raw_dir = Path("data/biographies")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"{args.person.lower().replace(' ', '_')}.txt"
        with open(raw_file, "w") as f:
            f.write(bio_text)
        console.print(f"[dim]Saved raw biography to {raw_file}[/dim]")
    else:
        parser.print_help()
        console.print("\n[yellow]Example usage:[/yellow]")
        console.print("  uv run biography_extractor.py --person 'Richard Feynman' --auto")
        console.print("  uv run biography_extractor.py --input my_bio.txt --person 'John Doe'")
        console.print("  uv run biography_extractor.py --all")
        return

    analysis = extract_decision_points(bio_text, person_name)
    display_analysis(analysis)
    save_analysis(analysis)


if __name__ == "__main__":
    main()
