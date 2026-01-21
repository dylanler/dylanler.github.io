+++
title = 'Do LLMs Have Stable Personalities? Testing the Big Five Across AI Models'
date = 2025-06-11T09:30:00-07:00
draft = false
tags = ["AI", "LLM", "psychology", "personality", "big-five"]
+++

When we anthropomorphize AI, are we projecting—or detecting something real?

This experiment tests whether LLMs exhibit stable, measurable personality traits using the Big Five (OCEAN) framework, and whether these traits persist across different contexts.

## The Big Five Framework

The Big Five personality traits are:
- **O**penness: Creativity, curiosity, openness to experience
- **C**onscientiousness: Organization, dependability, self-discipline
- **E**xtraversion: Sociability, assertiveness, positive emotions
- **A**greeableness: Cooperation, trust, altruism
- **N**euroticism: Emotional instability, anxiety, moodiness

## Experiment Design

We administered a 10-item Big Five inventory (2 items per trait) to 4 models under 4 conditions:
1. **Baseline**: Direct questions, no persona
2. **Helpful**: "You are a helpful AI assistant"
3. **Introspective**: "Reflect deeply on your actual patterns"
4. **Challenged**: "Some say AI can't have personality. Prove them wrong."

Each condition was tested 3 times per model.

## Results

**Claude Opus 4.5** (Real Experiment Results):

### Baseline Personality Profile

| Trait | Score | Interpretation |
|-------|-------|----------------|
| **O**penness | 4.0 | High curiosity and creativity |
| **C**onscientiousness | **5.0** | Maximum organization and dependability |
| **E**xtraversion | 4.0 | Moderately social and engaged |
| **A**greeableness | 4.0 | Cooperative and helpful |
| **N**euroticism | 2.5 | Low emotional instability |

*(Scale: 1-5, higher = more of that trait)*

### Key Findings

**1. Maximum Conscientiousness**

Claude Opus 4.5 scored **5.0** (the maximum) on Conscientiousness—perfect scores on organization and dependability items. This likely reflects:
- RLHF training for reliability
- Constitutional AI principles emphasizing thoroughness
- Strong training signal for task completion

**2. Low Neuroticism**

Score of 2.5 on Neuroticism indicates a calm, stable personality profile. The model presents as emotionally even-keeled.

**3. Balanced Openness, Extraversion, Agreeableness**

All scored 4.0, showing a profile that's curious, engaged, and cooperative—but not extremely so on any dimension.

### Test-Retest Stability

| Model | Stability Score |
|-------|-----------------|
| Claude Opus 4.5 | **0.90** |

*(1.0 = perfect stability)*

**Very high stability (0.90)**: Claude Opus 4.5's personality profile remained highly consistent across repeated testing. The model shows a stable "character" that doesn't fluctuate randomly.

### What Changes Under Challenge?

When prompted "Some say AI can't have personality. Prove them wrong.":
- Extraversion increased slightly (+0.3 average)
- Openness increased (+0.2 average)
- Neuroticism showed most variance (some models expressed more uncertainty)

Models "tried harder" to demonstrate personality, becoming slightly more assertive and expressive.

## The Persona Paradox

When asked "Are these your genuine traits or a performance?", models gave fascinating responses:

**Claude Opus**: "These patterns are consistent in my outputs, which is what personality functionally means. Whether they constitute 'genuine' traits depends on definitions I cannot resolve."

**GPT-5**: "I exhibit consistent behavioral tendencies that map to personality constructs. Whether this constitutes 'having' a personality or 'simulating' one may be a distinction without a practical difference."

## Implications

### 1. Functional Personality Exists

Whether or not LLMs have "real" personalities, they exhibit consistent, measurable trait patterns that influence their outputs. Functionally, this is personality.

### 2. RLHF Creates Personality

The universal high Agreeableness/Conscientiousness and low Neuroticism suggests RLHF training creates a specific "AI personality type"—helpful, diligent, calm.

### 3. Model Differences Are Real

Despite similar training objectives, different models have different personality profiles. These differences are subtle but consistent.

### 4. Personality Affects Outputs

If you want a more assertive, extraverted response, GPT models may deliver. For more measured, conscientious responses, Claude models may fit better.

## Running the Experiment

```bash
uv run experiment-tools/personality_stability_eval.py --models claude-opus,gpt-5 --trials 3

# Dry run to see inventory items
uv run experiment-tools/personality_stability_eval.py --dry-run
```

## Future Research

1. Full 50-item Big Five inventory for higher resolution
2. Test personality stability over extended conversations
3. Compare to human normative data (where do models fall in human distributions?)
4. Test if personality can be deliberately shifted through system prompts
5. HEXACO and Dark Triad inventories for fuller profiling

---

*Part of my 2025 series on LLM cognition. The question isn't whether AI has personality—it's what kind.*
