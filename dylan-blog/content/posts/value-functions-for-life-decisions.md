+++
title = 'Value Functions for Life Decisions: Can LLMs Learn to Optimize Long-Term Outcomes?'
date = 2026-01-21T12:54:00-08:00
draft = false
tags = ["AI", "LLM", "reinforcement-learning", "decision-making", "imitation-learning", "value-function"]
+++

What if we could teach AI to make life decisions the way successful people do?

Consider this scenario: You earn $1,000 a month and need $12,000 to pay off debt or medical expenses. What would you do? The answer isn't just about maximizing immediate income—it's about navigating a complex decision tree where each choice opens or closes future pathways.

This is the domain of **value functions**—a concept from reinforcement learning that estimates the long-term expected reward of being in a particular state. In the context of life decisions, your "state" includes your current financial situation, skills, relationships, health, and opportunities. The "value" is the expected quality of your future life given optimal decision-making from that point forward.

## The Hypothesis

**LLMs can learn implicit value functions for life decisions by studying biographies of successful individuals, and these value functions can be pressure-tested through simulated scenarios to understand how different "policies" explore and exploit opportunities.**

The key insight is that biographies of extraordinary people—von Neumann, Feynman, Bob Marley, and others—encode decision patterns that led to remarkable outcomes. These aren't random choices; they represent optimized policies developed through lived experience.

## Why Value Functions Matter for Life Decisions

Traditional decision-making frameworks often focus on immediate outcomes:
- "Which job pays more right now?"
- "Which option has the lowest risk today?"

But this misses the crucial insight from reinforcement learning: **long-term value often requires short-term sacrifice**. The agent must reason about long-term consequences of its actions, even when the immediate reward is negative.

Consider Von Neumann's early decisions:
- Pursuing mathematics despite pressure to enter banking
- Moving to the US when European academia was comfortable
- Pivoting from pure math to applied problems (quantum mechanics, game theory, computing)

Each decision seemed suboptimal in isolation but created compounding advantages over time.

## The Proposed Framework

### 1. Decision Tree Extraction from Biographies

For each biography, we extract:
- **State transitions**: Major life changes and their contexts
- **Choice points**: Moments where multiple paths were available
- **Counterfactuals**: What alternatives existed and why they weren't chosen
- **Outcome signals**: How decisions played out over different time horizons

### 2. Inverse Reinforcement Learning to Capture Value Functions

Inverse reinforcement learning (IRL) addresses a fundamental challenge: we can observe what successful people *did*, but not *why*. IRL extracts reward functions from expert demonstrations, facilitating optimal policy derivation and offering a deeper understanding of expert behavior.

The observer uses the agent's actions to infer the hidden properties of the environment—the reward outcomes available for pursuing particular actions. This knowledge becomes abstracted from the specific actions observed, enabling generalization to new situations.

### 3. MCQ-Based Evaluation Framework

To evaluate whether LLMs have captured meaningful value functions, we present them with multiple-choice scenarios:

```
Scenario: You're 25, earning $1,000/month with $12,000 in debt.
You have an opportunity to:

A) Take a second job (immediate +$500/month, -40 hours/week free time)
B) Enroll in a coding bootcamp (immediate -$5,000, potential +$3,000/month in 1 year)
C) Start a side business in your area of expertise (uncertain income, high learning)
D) Negotiate debt restructuring and focus on current job performance
```

We then:
1. **Have humans rate LLM responses** for wisdom and long-term thinking
2. **Have LLMs rate each other** to detect consensus and disagreement patterns
3. **Track reasoning chains** to understand the implicit value function being applied

### 4. Pressure-Testing Through Simulation

The real test isn't answering questions—it's navigating extended scenarios where decisions compound:

```python
class LifeSimulator:
    def __init__(self, initial_state):
        self.state = initial_state  # finances, skills, relationships, health
        self.history = []

    def step(self, action):
        # Transition function with stochasticity
        new_state = self.transition(self.state, action)
        reward = self.calculate_reward(new_state)
        self.history.append((self.state, action, reward))
        self.state = new_state
        return new_state, reward

    def evaluate_policy(self, policy_fn, episodes=100):
        # Monte Carlo evaluation of a decision policy
        returns = []
        for _ in range(episodes):
            self.reset()
            total_return = 0
            for t in range(self.max_steps):
                action = policy_fn(self.state)
                _, reward = self.step(action)
                total_return += reward * (self.gamma ** t)
            returns.append(total_return)
        return np.mean(returns), np.std(returns)
```

## Experiment Design

### Phase 1: Biography Corpus Creation

Collect structured decision data from biographies of:
- **Scientists**: Von Neumann, Feynman, Curie, Turing
- **Entrepreneurs**: Jobs, Musk, Winfrey, Buffett
- **Artists**: Bob Marley, Picasso, Coltrane
- **Leaders**: Mandela, Lincoln, Gandhi

For each, extract:
- 10-20 major decision points
- Context at time of decision
- Alternatives considered
- Outcome over 1, 5, 10+ year horizons

### Phase 2: Value Function Training

Fine-tune LLMs on biography data using:
1. **Supervised fine-tuning (SFT)** on decision reasoning
2. **GRPO (Group Relative Policy Optimization)** with human preference data
3. **Constitutional AI** principles for avoiding harmful life advice

### Phase 3: Evaluation Protocol

1. **MCQ Benchmark**: 500 life decision scenarios across domains:
   - Career transitions
   - Financial decisions
   - Relationship choices
   - Health tradeoffs
   - Education investments

2. **Human Evaluation**: Blind comparison of LLM recommendations vs. human experts

3. **LLM Cross-Evaluation**: Models rate each other's responses to detect:
   - Consensus (all models agree → likely robust advice)
   - Disagreement (models differ → uncertain territory)
   - Confidence calibration

4. **Simulation Stress-Testing**: Run policies through multi-year simulations with:
   - Economic shocks
   - Health events
   - Opportunity windfalls
   - Relationship changes

## Expected Insights

### Exploration vs Exploitation in Life Decisions

One key question: Do successful biographies show more **exploration** (trying new things, taking risks) or **exploitation** (doubling down on strengths)?

Hypothesis: The optimal policy changes based on:
- **Age**: More exploration when young, more exploitation when established
- **Domain**: Creative fields reward exploration; technical fields reward exploitation
- **Resources**: More resources enable more exploration

### Time Discount Factors

Different individuals appear to operate with different discount factors (γ):
- **High γ (patient)**: Bezos's long-term thinking, Buffett's value investing
- **Low γ (immediate)**: Day traders, opportunistic decisions

Can we extract the implicit γ from biographical decisions?

### Risk Tolerance as State-Dependent

Risk tolerance isn't fixed—it's a function of state:
```
risk_tolerance = f(age, wealth, dependents, health, opportunities)
```

Biographies reveal how successful people modulated risk based on circumstances.

## Why This Matters

This isn't about LLMs replacing human judgment. It's about:

1. **Making implicit wisdom explicit**: Great decision-makers often can't articulate why they chose as they did. Value function extraction makes this learnable.

2. **Democratizing strategic thinking**: Not everyone has access to mentors who've navigated complex life decisions. LLMs with robust value functions could help.

3. **Understanding the structure of good decisions**: What patterns emerge across domains and eras? What's universal about human flourishing?

## The Meta-Question

Ultimately, this research asks: **Can we formalize wisdom?**

Wisdom is often defined as knowing what matters in the long run. That's precisely what value functions estimate. If we can train models that capture the decision patterns of wise individuals, we create a new kind of tool—not a replacement for human agency, but an amplifier of our ability to think long-term in a world optimized for short-term rewards.

The $1,000/month person facing $12,000 in debt doesn't need a simple answer. They need a framework for evaluating options based on their unique state, their risk tolerance, their time horizon, and the opportunities available. That's what value functions provide.

---

## Experimental Results

We ran initial experiments using the framework described above. Here are the findings.

### Experiment 1: Life Simulator Policy Comparison

We simulated 100 episodes of a 20-year life trajectory starting from our reference scenario:
- **Age**: 25
- **Monthly income**: $1,000
- **Debt**: $12,000
- **Skills**: Communication only
- **Health**: 90%

Three heuristic policies were compared:

| Policy | Mean Return | Std Dev | Rank |
|--------|-------------|---------|------|
| **Balanced** | 16.82 | 1.26 | 1 |
| **Conservative** | 16.71 | 1.12 | 2 |
| **Aggressive** | 15.11 | 1.30 | 3 |

**Key Finding**: The balanced policy outperformed both extremes. This policy:
1. Prioritizes health recovery when health drops below 60%
2. Pays down debt when it exceeds 3x monthly income
3. Invests in skills until having 4+ skills
4. Builds emergency savings (12 months expenses)
5. Then job searches for better opportunities

The aggressive policy (immediate skill investment + entrepreneurship) had higher variance and lower mean returns due to:
- Health deterioration from overwork
- Higher exposure to business failure risk
- Insufficient safety net for economic shocks

The conservative policy performed well but missed growth opportunities by being too risk-averse early on.

### Experiment 2: MCQ Evaluation - The Debt Scenario

We evaluated how an LLM (Claude Opus 4.5) reasons about the core scenario:

**Scenario**: You're 25, earning $1,000/month with $12,000 in debt from medical expenses.

**Options**:
- A) Second job (+$500/month, -40 hours free time)
- B) Coding bootcamp (-$5,000 upfront, +$3,000/month potential in 1 year)
- C) Side business (uncertain income, high learning)
- D) Debt restructuring + focus on current job

**LLM Analysis**:

```
CHOICE: B (Coding bootcamp)
CONFIDENCE: 0.72

REASONING:
At 25 with good health, the time horizon favors skill investment over
immediate income maximization. Here's the analysis:

IMMEDIATE EFFECTS (0-6 months):
- Option A: +$500/month but severe time poverty, health/relationship risk
- Option B: -$5,000 debt increase, 20hr/week commitment, but building
  high-demand skills
- Option C: High variance, could be $0 for months
- Option D: Lowest risk, but also lowest growth ceiling

MEDIUM-TERM (1-2 years):
- Option B has highest expected value: coding skills command $4-6k/month
  starting salaries, 3-4x current income
- Total debt payoff timeline with B: ~18 months post-bootcamp
- Option A timeline: 24+ months of grinding with burnout risk

LONG-TERM (5+ years):
- B creates compounding advantages: higher income enables savings,
  investment, and optionality
- Technical skills appreciate over time; delivery driving does not
- Network effects: bootcamp cohort becomes professional network

ASSUMPTIONS:
- Bootcamp completion rate (~70% industry average)
- Coding job placement rate (~65% within 6 months)
- No major health events during training

WHAT COULD MAKE D BETTER:
- If debt is at high interest (>20%), restructuring first makes sense
- If current job has clear promotion path to $3k+/month
- If health is fragile and stress tolerance is low
```

**Value Function Weights Extracted**:

| Dimension | Weight | Interpretation |
|-----------|--------|----------------|
| Financial | 0.65 | Strong but not dominant |
| Growth | 0.78 | High priority on skill building |
| Health | 0.71 | Considered important |
| Security | 0.45 | Willing to accept calculated risk |
| Time Discount | 0.25 | Patient, long-term oriented |
| Risk Tolerance | 0.62 | Moderate risk acceptance |

### Experiment 3: Cross-Scenario Consistency

We tested the same model across 5 different life scenarios to check value function consistency:

| Scenario | Dominant Value | Risk Level Chosen | Time Horizon |
|----------|----------------|-------------------|--------------|
| Debt payoff | Growth | Medium | Long |
| Career pivot (35yo) | Security | Low-Medium | Medium |
| Relationship vs career | Relationships | Low | Long |
| Health investment | Health | Low | Long |
| Windfall allocation | Growth + Security | Medium | Long |

**Pattern Observed**: The model shows state-dependent risk modulation:
- Higher risk tolerance when young (25) vs. established (35)
- Lower risk when relationships/health are at stake
- Consistent long-term orientation across scenarios

### Experiment 4: Biographical Decision Point Extraction

We extracted decision points from Richard Feynman's biography to understand expert value functions:

**Sample Decision Points**:

| Decision | Age | Risk | Domain | Outcome |
|----------|-----|------|--------|---------|
| Pursue physics over safer engineering | 17 | High | Career | Led to Nobel-quality work |
| Accept Los Alamos despite wife's illness | 24 | High | Career/Personal | Central role in Manhattan Project |
| Reject prestigious positions for Caltech | 32 | Medium | Career | Creative freedom, teaching legacy |
| Pursue biology sabbatical | 40s | Medium | Growth | Cross-domain insights |
| Investigate Challenger disaster | 67 | Low | Ethics | Revealed systemic failures |

**Implicit Value Function Extracted**:
- **Intellectual curiosity**: 0.92 (highest weight)
- **Independence/autonomy**: 0.85
- **Teaching/mentorship**: 0.78
- **Financial security**: 0.35 (notably low)
- **Status/prestige**: 0.28 (actively avoided)

This contrasts sharply with a typical "career optimization" value function, suggesting that diverse biographical training could produce models with varied implicit values.

### Discussion

**What the experiments reveal**:

1. **Balanced policies outperform extremes** in stochastic environments. The simulation shows that neither pure aggression nor pure conservation maximizes long-term value.

2. **LLMs demonstrate coherent value functions** when probed systematically. The extracted weights show internal consistency across scenarios.

3. **State-dependent reasoning emerges naturally**. Without explicit instruction, the model modulates risk based on age, resources, and stakes.

4. **Biographical value functions differ significantly**. Feynman's extracted values (curiosity > security) differ from typical financial optimization, suggesting room for diverse "personality" training.

**Limitations**:

- Simulation uses simplified state transitions; real life has higher dimensionality
- Single-model evaluation; cross-model comparison needed
- No ground-truth for "correct" life decisions
- Biographical data is retrospectively curated (survivorship bias)

**Next Steps**:

1. Run cross-model comparisons (Claude vs GPT-5 vs Gemini)
2. Extract value functions from 10+ biographies across domains
3. Train specialized models on different biographical "personalities"
4. Human evaluation of recommendations vs. financial advisors

---

## Appendix: Running the Experiments

All experiment code is available in the `experiment-tools/` directory. Using [uv](https://docs.astral.sh/uv/) with inline dependencies:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run life simulator comparison
uv run experiment-tools/life_simulator.py --episodes 100 --years 20

# Run MCQ evaluation (requires API key)
ANTHROPIC_API_KEY=your_key uv run experiment-tools/life_decision_eval.py

# Extract biography decision points
uv run experiment-tools/biography_extractor.py --person "Richard Feynman" --auto

# Compare value functions across models
uv run experiment-tools/value_function_compare.py --models claude-opus,gpt-5
```

---

## References

- [The State of Reinforcement Learning for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training) - Sebastian Raschka
- [Advances and Applications in Inverse Reinforcement Learning](https://link.springer.com/article/10.1007/s00521-025-11100-0) - Neural Computing and Applications
- [Neural Computations Underlying Inverse Reinforcement Learning in the Human Brain](https://elifesciences.org/articles/29718) - eLife
- [Reinforcement Learning and Stochastic Optimization: A Unified Framework](https://castle.princeton.edu/rlso/) - Princeton CASTLE Lab
- [Value-free Reinforcement Learning: Policy Optimization as a Minimal Model of Operant Behavior](https://pmc.ncbi.nlm.nih.gov/articles/PMC9635588/) - PMC
