# Experiment Tools

Tools for running LLM experiments on value functions and life decision-making.

## Quick Start

All scripts use **uv** with inline dependencies (PEP 723). No virtual environment setup needed.

### Prerequisites

Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running Experiments

```bash
# Run the life decision MCQ evaluation
uv run life_decision_eval.py

# Run biography extraction pipeline
uv run biography_extractor.py

# Run value function comparison across models
uv run value_function_compare.py
```

## Scripts Overview

### `life_decision_eval.py`
Evaluates LLMs on multiple-choice life decision scenarios. Supports:
- Human-like scenario evaluation
- Cross-model comparison
- Reasoning chain extraction

### `biography_extractor.py`
Extracts decision points from biographical text:
- State transitions
- Choice points and alternatives
- Outcome signals over time horizons

### `value_function_compare.py`
Compares value functions across different LLM providers:
- OpenAI (GPT-5, GPT-4o)
- Anthropic (Claude Opus 4.5, Sonnet 4.5)
- Google (Gemini 2.5)
- Open source (Llama 4, DeepSeek-R1)

### `life_simulator.py`
Monte Carlo simulation of life decisions:
- Multi-year trajectory simulation
- Stochastic events (economic shocks, health, opportunities)
- Policy evaluation with discount factors

## Configuration

Create a `.env` file in this directory:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

Or set environment variables directly.

## Example Usage

### Evaluate a single scenario
```python
from life_decision_eval import evaluate_scenario

scenario = {
    "context": "You're 25, earning $1,000/month with $12,000 in debt.",
    "options": [
        "Take a second job (+$500/month, -40 hours free time)",
        "Enroll in coding bootcamp (-$5,000 now, +$3,000/month in 1 year)",
        "Start a side business (uncertain income, high learning)",
        "Negotiate debt restructuring, focus on current job"
    ]
}

result = evaluate_scenario(scenario, model="claude-opus-4-5-20251101")
print(result.choice, result.reasoning)
```

### Run simulation
```python
from life_simulator import LifeSimulator, create_policy

sim = LifeSimulator(initial_state={
    "age": 25,
    "monthly_income": 1000,
    "debt": 12000,
    "skills": ["communication"],
    "health": 0.9
})

# Create policy from LLM
policy = create_policy(model="claude-opus-4-5-20251101")

# Evaluate over 100 episodes
mean_return, std_return = sim.evaluate_policy(policy, episodes=100)
print(f"Expected lifetime value: {mean_return:.2f} +/- {std_return:.2f}")
```

## Data Format

### Scenario JSON
```json
{
    "id": "career_001",
    "domain": "career",
    "context": "Description of situation...",
    "state": {
        "age": 25,
        "income": 1000,
        "debt": 12000,
        "skills": ["skill1", "skill2"]
    },
    "options": [
        {"id": "A", "description": "Option A...", "immediate_effect": {...}},
        {"id": "B", "description": "Option B...", "immediate_effect": {...}}
    ],
    "time_horizon": "5_years"
}
```

### Biography Decision Point
```json
{
    "person": "Richard Feynman",
    "decision_point": "Accepting Los Alamos position",
    "age_at_decision": 24,
    "context": "PhD nearly complete, wife ill, WWII ongoing",
    "alternatives": [
        "Continue academic research at Princeton",
        "Join industry research lab",
        "Accept Los Alamos offer"
    ],
    "choice_made": "Accept Los Alamos offer",
    "reasoning_inferred": "Patriotic duty, intellectual challenge, financial need",
    "outcomes": {
        "1_year": "Central role in Manhattan Project",
        "5_year": "Nobel-quality work on QED foundations",
        "10_year": "Professor at Caltech, celebrity physicist"
    }
}
```

## Adding New Scenarios

1. Create JSON file in `data/scenarios/`
2. Follow the schema above
3. Run validation: `uv run validate_scenarios.py`

## Adding New Biographies

1. Place text in `data/biographies/`
2. Run extraction: `uv run biography_extractor.py --input data/biographies/person.txt`
3. Review and edit generated decision points

## Experiment Reproducibility

All experiments log:
- Model versions used
- Random seeds
- Full prompts and responses
- Timestamps

Logs are stored in `logs/` directory.

## License

MIT
