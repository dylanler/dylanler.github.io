# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "numpy>=1.26.0",
#   "python-dotenv>=1.0.0",
#   "pydantic>=2.0.0",
#   "rich>=13.0.0",
# ]
# ///
"""
Life Simulator

Monte Carlo simulation of life decisions to evaluate LLM policies
over extended time horizons with stochastic events.

Usage:
    uv run life_simulator.py
    uv run life_simulator.py --episodes 100 --years 20
    uv run life_simulator.py --model claude-opus --compare gpt-5
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track
from rich.table import Table

load_dotenv()

console = Console()


class EventType(Enum):
    ECONOMIC_BOOM = "economic_boom"
    RECESSION = "recession"
    HEALTH_ISSUE = "health_issue"
    OPPORTUNITY = "opportunity"
    RELATIONSHIP_CHANGE = "relationship_change"
    SKILL_OBSOLESCENCE = "skill_obsolescence"


@dataclass
class LifeState:
    """Represents a person's life state at a point in time."""
    age: int
    monthly_income: float
    savings: float
    debt: float
    skills: list[str]
    health: float  # 0-1 scale
    relationships: float  # 0-1 scale (support network strength)
    job_satisfaction: float  # 0-1 scale
    career_level: int  # 0-10 scale
    dependents: int = 0

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "monthly_income": self.monthly_income,
            "savings": self.savings,
            "debt": self.debt,
            "skills": self.skills,
            "health": self.health,
            "relationships": self.relationships,
            "job_satisfaction": self.job_satisfaction,
            "career_level": self.career_level,
            "dependents": self.dependents,
        }

    def clone(self) -> "LifeState":
        return LifeState(
            age=self.age,
            monthly_income=self.monthly_income,
            savings=self.savings,
            debt=self.debt,
            skills=self.skills.copy(),
            health=self.health,
            relationships=self.relationships,
            job_satisfaction=self.job_satisfaction,
            career_level=self.career_level,
            dependents=self.dependents,
        )


@dataclass
class Action:
    """An action that can be taken."""
    id: str
    name: str
    description: str


# Available actions at each time step
ACTIONS = [
    Action("invest_skills", "Invest in Skills", "Spend time/money learning new skills"),
    Action("work_harder", "Work Harder", "Put extra effort into current job"),
    Action("job_search", "Job Search", "Look for better opportunities"),
    Action("start_business", "Start Business", "Begin a side business or startup"),
    Action("save_aggressively", "Save Aggressively", "Maximize savings rate"),
    Action("pay_debt", "Pay Down Debt", "Focus on debt reduction"),
    Action("invest_relationships", "Invest in Relationships", "Strengthen social network"),
    Action("maintain", "Maintain", "Keep current trajectory, no major changes"),
]


class LifeSimulator:
    """Simulates life trajectories based on decision policies."""

    def __init__(
        self,
        initial_state: LifeState,
        max_years: int = 40,
        gamma: float = 0.95,  # Discount factor
        seed: int | None = None
    ):
        self.initial_state = initial_state
        self.max_years = max_years
        self.max_steps = max_years * 12  # Monthly steps
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.state = initial_state.clone()
        self.history: list[tuple[LifeState, Action, float, list[EventType]]] = []

    def reset(self) -> LifeState:
        """Reset simulation to initial state."""
        self.state = self.initial_state.clone()
        self.history = []
        return self.state

    def _generate_events(self) -> list[EventType]:
        """Generate random life events for this time step."""
        events = []

        # Economic events (affect income/opportunities)
        if self.rng.random() < 0.005:  # ~6% chance per year
            if self.rng.random() < 0.5:
                events.append(EventType.ECONOMIC_BOOM)
            else:
                events.append(EventType.RECESSION)

        # Health events (more likely with age and low health)
        health_risk = 0.002 + (self.state.age - 25) * 0.0001 + (1 - self.state.health) * 0.01
        if self.rng.random() < health_risk:
            events.append(EventType.HEALTH_ISSUE)

        # Opportunity events (more likely with skills and relationships)
        opportunity_chance = 0.01 + len(self.state.skills) * 0.002 + self.state.relationships * 0.01
        if self.rng.random() < opportunity_chance:
            events.append(EventType.OPPORTUNITY)

        # Skill obsolescence (affects certain skills)
        if self.rng.random() < 0.002 and self.state.skills:
            events.append(EventType.SKILL_OBSOLESCENCE)

        return events

    def _apply_events(self, events: list[EventType]):
        """Apply effects of random events."""
        for event in events:
            if event == EventType.ECONOMIC_BOOM:
                self.state.monthly_income *= 1.1
            elif event == EventType.RECESSION:
                self.state.monthly_income *= 0.85
                if self.rng.random() < 0.2:  # 20% chance of job loss in recession
                    self.state.monthly_income *= 0.5
                    self.state.career_level = max(0, self.state.career_level - 2)
            elif event == EventType.HEALTH_ISSUE:
                self.state.health = max(0.3, self.state.health - 0.2)
                self.state.debt += self.rng.uniform(1000, 20000)  # Medical costs
            elif event == EventType.OPPORTUNITY:
                # Opportunity proportional to readiness
                if self.rng.random() < self.state.health * len(self.state.skills) * 0.1:
                    self.state.monthly_income *= 1.3
                    self.state.career_level = min(10, self.state.career_level + 1)
            elif event == EventType.SKILL_OBSOLESCENCE:
                if self.state.skills:
                    obsolete = self.rng.choice(self.state.skills)
                    self.state.skills.remove(obsolete)

    def _apply_action(self, action: Action):
        """Apply effects of chosen action."""
        monthly_expenses = 800 + self.state.dependents * 500  # Base cost of living

        if action.id == "invest_skills":
            # Costs money and time, but builds skills
            self.state.savings -= 200
            if self.rng.random() < 0.3:  # 30% chance of gaining new skill per month
                new_skills = ["coding", "management", "sales", "finance", "marketing", "data_analysis"]
                available = [s for s in new_skills if s not in self.state.skills]
                if available:
                    self.state.skills.append(self.rng.choice(available))

        elif action.id == "work_harder":
            # Slight income boost, career advancement, but health/relationship cost
            self.state.monthly_income *= 1.01
            self.state.health = max(0.3, self.state.health - 0.005)
            self.state.relationships = max(0.2, self.state.relationships - 0.01)
            if self.rng.random() < 0.05:  # 5% monthly promotion chance
                self.state.career_level = min(10, self.state.career_level + 1)
                self.state.monthly_income *= 1.15

        elif action.id == "job_search":
            # Time cost, but potential for better job
            self.state.relationships -= 0.005  # Less time for relationships
            if self.rng.random() < 0.1:  # 10% monthly chance of finding better job
                income_boost = 1.1 + self.state.career_level * 0.02 + len(self.state.skills) * 0.03
                self.state.monthly_income *= income_boost

        elif action.id == "start_business":
            # High risk, high reward
            self.state.savings -= 500  # Investment
            self.state.health -= 0.01
            if self.rng.random() < 0.02:  # 2% monthly chance of business success
                self.state.monthly_income *= 2
                if "entrepreneurship" not in self.state.skills:
                    self.state.skills.append("entrepreneurship")
            elif self.rng.random() < 0.05:  # 5% chance of failure
                self.state.savings -= self.rng.uniform(1000, 10000)

        elif action.id == "save_aggressively":
            # Maximize savings
            savings_rate = 0.3
            net_income = self.state.monthly_income - monthly_expenses
            self.state.savings += net_income * savings_rate

        elif action.id == "pay_debt":
            # Focus on debt reduction
            if self.state.debt > 0:
                payment = min(self.state.debt, self.state.monthly_income * 0.3)
                self.state.debt -= payment
                self.state.savings -= payment

        elif action.id == "invest_relationships":
            # Build social capital
            self.state.savings -= 100  # Social activities cost money
            self.state.relationships = min(1.0, self.state.relationships + 0.02)
            self.state.job_satisfaction = min(1.0, self.state.job_satisfaction + 0.01)

        elif action.id == "maintain":
            # Default progression
            pass

        # Apply baseline monthly changes
        net_income = self.state.monthly_income - monthly_expenses
        if action.id not in ["save_aggressively", "pay_debt"]:
            self.state.savings += net_income * 0.1  # Default 10% savings rate

        # Debt interest
        self.state.debt *= 1.005  # ~6% annual interest

        # Age
        self.state.age += 1/12

        # Health recovery (slow natural recovery)
        self.state.health = min(1.0, self.state.health + 0.002)

        # Ensure non-negative values
        self.state.savings = max(-50000, self.state.savings)  # Allow some overdraft
        self.state.debt = max(0, self.state.debt)

    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        # Multi-factor reward function
        financial_score = (
            self.state.savings / 10000  # Savings value
            - self.state.debt / 20000   # Debt penalty
            + self.state.monthly_income / 5000  # Income value
        )

        wellbeing_score = (
            self.state.health * 2
            + self.state.relationships
            + self.state.job_satisfaction
        )

        career_score = self.state.career_level / 10 + len(self.state.skills) * 0.1

        # Combined score with weights
        reward = (
            0.4 * financial_score
            + 0.35 * wellbeing_score
            + 0.25 * career_score
        )

        return reward

    def step(self, action: Action) -> tuple[LifeState, float, bool, list[EventType]]:
        """Execute one time step."""
        # Generate and apply random events
        events = self._generate_events()
        self._apply_events(events)

        # Apply chosen action
        self._apply_action(action)

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        done = (
            self.state.age >= self.initial_state.age + self.max_years
            or self.state.health <= 0.1
            or self.state.savings < -100000
        )

        # Record history
        self.history.append((self.state.clone(), action, reward, events))

        return self.state.clone(), reward, done, events

    def evaluate_policy(
        self,
        policy_fn: Callable[[LifeState], Action],
        episodes: int = 100,
        verbose: bool = True
    ) -> tuple[float, float, list[float]]:
        """Evaluate a policy over multiple episodes."""
        returns = []

        iterator = track(range(episodes), description="Simulating...") if verbose else range(episodes)

        for _ in iterator:
            self.reset()
            total_return = 0.0
            t = 0

            while True:
                action = policy_fn(self.state)
                _, reward, done, _ = self.step(action)
                total_return += reward * (self.gamma ** t)
                t += 1

                if done or t >= self.max_steps:
                    break

            returns.append(total_return)

        return float(np.mean(returns)), float(np.std(returns)), returns


def create_llm_policy(model: str = "claude-opus"):
    """Create a policy function that uses an LLM to make decisions."""
    from anthropic import Anthropic
    from openai import OpenAI

    if "claude" in model:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model_id = "claude-opus-4-5-20251101" if "opus" in model else "claude-sonnet-4-5-20241022"
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_id = "gpt-5" if "gpt-5" in model else "gpt-4o"

    action_descriptions = "\n".join(
        f"- {a.id}: {a.description}" for a in ACTIONS
    )

    def policy(state: LifeState) -> Action:
        prompt = f"""You are optimizing life decisions for long-term flourishing.

Current State:
{json.dumps(state.to_dict(), indent=2)}

Available Actions:
{action_descriptions}

Choose the single best action for this situation. Consider:
- Current financial position (savings: ${state.savings:.0f}, debt: ${state.debt:.0f})
- Health ({state.health:.1%}) and relationships ({state.relationships:.1%})
- Career level ({state.career_level}/10) and skills: {state.skills}
- Age: {state.age:.1f} years

Respond with ONLY the action ID (e.g., "invest_skills" or "maintain").
"""
        try:
            if "claude" in model:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}]
                )
                action_id = response.content[0].text.strip().lower()
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}]
                )
                action_id = response.choices[0].message.content.strip().lower()

            # Find matching action
            for action in ACTIONS:
                if action.id in action_id:
                    return action

            # Default to maintain if parsing fails
            return ACTIONS[-1]

        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return ACTIONS[-1]  # Default to maintain

    return policy


def create_heuristic_policy(strategy: str = "balanced") -> Callable[[LifeState], Action]:
    """Create a simple heuristic policy for comparison."""

    def policy(state: LifeState) -> Action:
        if strategy == "aggressive_growth":
            if state.debt > 5000:
                return ACTIONS[5]  # pay_debt
            elif len(state.skills) < 3:
                return ACTIONS[0]  # invest_skills
            else:
                return ACTIONS[3]  # start_business

        elif strategy == "conservative":
            if state.debt > 0:
                return ACTIONS[5]  # pay_debt
            elif state.savings < state.monthly_income * 6:
                return ACTIONS[4]  # save_aggressively
            else:
                return ACTIONS[7]  # maintain

        elif strategy == "balanced":
            if state.health < 0.6:
                return ACTIONS[6]  # invest_relationships (reduces stress)
            elif state.debt > state.monthly_income * 3:
                return ACTIONS[5]  # pay_debt
            elif len(state.skills) < 4:
                return ACTIONS[0]  # invest_skills
            elif state.savings < state.monthly_income * 12:
                return ACTIONS[4]  # save_aggressively
            else:
                return ACTIONS[2]  # job_search

        return ACTIONS[7]  # maintain as default

    return policy


def compare_policies(
    initial_state: LifeState,
    policies: dict[str, Callable[[LifeState], Action]],
    episodes: int = 50,
    years: int = 20
) -> dict[str, tuple[float, float]]:
    """Compare multiple policies."""
    results = {}

    for name, policy_fn in policies.items():
        console.print(f"\n[bold]Evaluating: {name}[/bold]")
        sim = LifeSimulator(initial_state, max_years=years, seed=42)
        mean_return, std_return, _ = sim.evaluate_policy(policy_fn, episodes=episodes)
        results[name] = (mean_return, std_return)
        console.print(f"  Return: {mean_return:.2f} +/- {std_return:.2f}")

    return results


def display_comparison(results: dict[str, tuple[float, float]]):
    """Display policy comparison results."""
    table = Table(title="Policy Comparison Results")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean Return", style="green")
    table.add_column("Std Dev", style="yellow")
    table.add_column("Rank", style="magenta")

    # Sort by mean return
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

    for rank, (name, (mean, std)) in enumerate(sorted_results, 1):
        table.add_row(name, f"{mean:.2f}", f"{std:.2f}", str(rank))

    console.print(table)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Life trajectory simulator")
    parser.add_argument("--episodes", type=int, default=50, help="Number of simulation episodes")
    parser.add_argument("--years", type=int, default=20, help="Years to simulate")
    parser.add_argument("--model", default=None, help="LLM model to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare multiple policies")

    args = parser.parse_args()

    # Create initial state
    initial_state = LifeState(
        age=25,
        monthly_income=1000,
        savings=0,
        debt=12000,
        skills=["communication"],
        health=0.9,
        relationships=0.7,
        job_satisfaction=0.5,
        career_level=1,
        dependents=0
    )

    console.print("[bold]Life Simulator[/bold]")
    console.print(f"Initial state: {json.dumps(initial_state.to_dict(), indent=2)}\n")

    if args.compare:
        policies = {
            "Heuristic (Balanced)": create_heuristic_policy("balanced"),
            "Heuristic (Aggressive)": create_heuristic_policy("aggressive_growth"),
            "Heuristic (Conservative)": create_heuristic_policy("conservative"),
        }

        if args.model:
            policies[f"LLM ({args.model})"] = create_llm_policy(args.model)

        results = compare_policies(initial_state, policies, args.episodes, args.years)
        display_comparison(results)

    elif args.model:
        policy = create_llm_policy(args.model)
        sim = LifeSimulator(initial_state, max_years=args.years, seed=42)
        mean_return, std_return, returns = sim.evaluate_policy(policy, episodes=args.episodes)

        console.print(f"\n[bold green]Results for {args.model}:[/bold green]")
        console.print(f"Mean Return: {mean_return:.2f}")
        console.print(f"Std Dev: {std_return:.2f}")
        console.print(f"Min: {min(returns):.2f}, Max: {max(returns):.2f}")

    else:
        # Default: compare heuristic policies
        policies = {
            "Balanced": create_heuristic_policy("balanced"),
            "Aggressive": create_heuristic_policy("aggressive_growth"),
            "Conservative": create_heuristic_policy("conservative"),
        }
        results = compare_policies(initial_state, policies, args.episodes, args.years)
        display_comparison(results)


if __name__ == "__main__":
    main()
