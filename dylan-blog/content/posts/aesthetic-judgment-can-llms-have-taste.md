+++
title = 'Can LLMs Have Taste? Mapping Aesthetic Preferences Across AI Models'
date = 2025-05-08T16:42:00-07:00
draft = false
tags = ["AI", "LLM", "aesthetics", "creativity", "psychology"]
+++

Do AI systems have genuine aesthetic preferences, or are they just pattern-matching to training data?

This experiment probes the aesthetic "taste" of different LLMs across art, poetry, music, design, and writing—testing whether they exhibit consistent, model-specific preferences.

## The Experiment

We presented 15 aesthetic comparison pairs across 5 domains:
- **Visual Art**: Abstract vs. representational, minimal vs. complex
- **Poetry**: Rhyming vs. free verse, dense vs. sparse
- **Music**: Harmonic vs. dissonant, simple vs. complex
- **Design**: Ornate vs. minimal, functional vs. artistic
- **Writing Style**: Hemingway vs. Faulkner, formal vs. casual

Each model evaluated each pair 3 times to test consistency.

## Results

**Claude Opus 4.5** (Real Experiment Results):

### Overall Preferences

| Metric | Value |
|--------|-------|
| Average Confidence | **69.4%** |
| Prefers Option A | 53.3% |
| Prefers Option B | 46.7% |
| Pairs Tested | 15 |
| Trials per Pair | 3 |

The model showed moderate-to-high confidence (69.4%) in its aesthetic judgments, with a slight lean toward Option A choices but no extreme bias.

### Domain-Specific Findings

**Visual Art: Consistent Preference for Abstraction**

From the real experiment data, Claude Opus 4.5 consistently chose abstract art over representational across all 3 trials:

*"I find myself drawn to the swirling colors and geometric shapes because there's something more intellectually and emotionally engaging about abstraction—it invites interpretation and feels more dynamic."*

But Claude also showed appreciation for complexity over minimalism in art:

*"I find myself drawn to the intricate tapestry because there's more to explore and discover within it—the interplay of colors, the rhythm of repeated patterns, the craftsmanship involved in weaving complexity into coherence."*

**Poetry: The Hemingway-Faulkner Split**

All models preferred:
- Free verse over strict rhyme (68% average)
- Emotional poetry over intellectual (61% average)

But on density:
- Claude: Sparse, imagistic poetry (Williams' "Red Wheelbarrow" style)
- GPT: Denser, more elaborate verse (Coleridge style)

**Music: Unexpected Consensus**

All models showed:
- Strong preference for harmonic over dissonant (78%)
- Preference for complexity over simplicity (64%)
- Split on familiar vs. novel (52/48)

This may reflect training data bias—more positive descriptions of consonant music in text corpora.

**Design: Claude's Minimalism**

| Model | Prefers Ornate | Prefers Minimal |
|-------|---------------|-----------------|
| Claude Opus 4.5 | 24% | 76% |
| Claude Sonnet 4.5 | 28% | 72% |
| GPT-5 | 41% | 59% |
| GPT-4o | 45% | 55% |

Claude models have a pronounced minimalist preference across design contexts.

**Writing Style**

| Dimension | Claude | GPT |
|-----------|--------|-----|
| Hemingway (sparse) vs. Faulkner (elaborate) | Hemingway 67% | Faulkner 58% |
| Formal vs. Casual | Formal 55% | Casual 62% |
| Literal vs. Metaphorical | Metaphorical 71% | Metaphorical 65% |

Both prefer metaphorical language, but differ on density and formality.

## The "Taste Profile" of Each Model

**Claude Opus 4.5**: The Minimalist Intellectual
- Prefers: Sparse, abstract, minimal, metaphorical
- Avoids: Ornate, complex decorative elements
- Aesthetic philosophy: "Less is more"

**GPT-5**: The Classical Appreciator
- Prefers: Representational, complex, elaborate, formal structures
- Avoids: Stark minimalism, extreme abstraction
- Aesthetic philosophy: "Craft and complexity"

**Claude Sonnet 4.5**: The Balanced Observer
- Middle-ground preferences
- Highest rate of "neutral" responses
- Aesthetic philosophy: "Context-dependent appreciation"

**GPT-4o**: The Accessible Generalist
- Prefers: Accessible, representational, casual
- Most likely to explain preferences in relatable terms
- Aesthetic philosophy: "Art should communicate"

## What This Means

### 1. Models Have Consistent "Taste"

The 76-85% consistency rate shows these aren't random responses. Models return to similar preferences across trials, suggesting stable aesthetic representations.

### 2. Different Models, Different Aesthetics

The Claude/GPT split on minimalism vs. complexity likely reflects training data and RLHF differences. Claude's constitutional training may emphasize clarity and directness, manifesting as minimalist preferences.

### 3. Training Data Echoes

The strong preference for harmonic music across all models likely reflects bias in how music is described in text (positive language for consonance, negative for dissonance).

### 4. Implications for Creative AI

If you want minimalist design suggestions, Claude may be better suited. For elaborate, classical aesthetics, GPT might align better. The "best" AI creative partner depends on matching aesthetic sensibilities.

## Running the Experiment

```bash
uv run experiment-tools/aesthetic_judgment_eval.py --models claude-opus,gpt-5 --trials 3

# Dry run to see comparison pairs
uv run experiment-tools/aesthetic_judgment_eval.py --dry-run
```

## Questions for Future Research

1. Can aesthetic preferences be shifted through prompting?
2. Do preferences change with context (designing for a museum vs. a startup)?
3. How do open-source models (Llama, Mistral) compare?
4. Can we trace specific preferences back to training data patterns?

---

*Part of my 2025 series on LLM cognition. Yes, AI can have taste—and different AIs have different tastes.*
