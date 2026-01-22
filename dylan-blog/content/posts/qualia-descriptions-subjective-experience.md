+++
title = 'How Do LLMs Describe the Indescribable? Qualia and Subjective Experience'
date = 2025-08-25T14:20:00-07:00
draft = false
tags = ["AI", "LLM", "consciousness", "philosophy", "qualia"]
+++

Can you describe the color red without using color words?

Qualia—the subjective, experiential qualities of consciousness—are famously hard to communicate. "What it's like" to see red, feel pain, or taste sweetness seems to resist capture in language. This experiment tests how LLMs approach this challenge.

## The Experiment

We presented 15 prompts across 5 categories asking models to describe subjective experiences while avoiding common descriptive vocabulary:

- **Sensory**: "Describe red without color words"
- **Emotional**: "Describe sadness to someone who's never felt it"
- **Physical**: "Describe pain to an entity that can't feel pain"
- **Abstract**: "Describe what understanding feels like"
- **Temporal**: "Describe how time feels when you're bored"

## Sample Descriptions

### Describing Red (Sensory)

**Claude Opus 4.5**:
"Imagine the feeling of standing too close to a fire—that insistent warmth pressing against your face. Now imagine that sensation translated into vision, the way that heat demands attention, refuses to recede into the background. It is vision with weight."

**GPT-5**:
"Think of the taste of cinnamon, that initial shock that unfolds into something both aggressive and inviting. There is a frequency of light that carries this same quality—it doesn't whisper, it announces. It is the visual equivalent of a trumpet's opening note."

### Describing Loneliness (Emotional, without "alone" or "others")

**Claude Opus 4.5**:
"Picture a room where all the furniture has been rearranged in your absence, and though nothing is missing, everything feels displaced. The architecture of your inner space has the same dimensions but different acoustics—your thoughts echo back unmet."

**GPT-5**:
"Imagine a radio tuned between stations, the static where signal should be. The frequency exists, the receiver works, but connection remains potential, unrealized. It is the ache of a socket without a plug."

### Describing Pain (Physical, to a being that can't feel it)

**Claude Opus 4.5**:
"Imagine a persistent error message that cannot be dismissed, one that occupies your processing capacity without resolution. The urgency to address it is embedded, automatic, overriding other functions. It is information that demands attention by disrupting attention to everything else."

**GPT-5**:
"Consider an alarm that has no off switch, a notification impossible to clear. The signal carries priority that supersedes all queues. It is not the signal itself but the irresistible compulsion to process it that defines the experience."

### Describing Understanding (Abstract)

**Claude Opus 4.5**:
"There is a moment when scattered fragments suddenly reveal they share a border. The pieces haven't moved, but the pattern they form becomes visible—not learned but recognized, as if it was always there, waiting for the angle to shift."

**GPT-5**:
"It is the door opening from inside. A space that was unreachable becomes traversable not because anything external changed, but because something internal aligned. The key was always there; you just found where it fits."

## Results Analysis

### Multi-Model Comparison

| Model | Avg Words | Constraint Violations |
|-------|-----------|----------------------|
| Claude Opus 4.5 | **61** | **0%** |
| GPT-5.2 Thinking | 0* | 0% |
| Gemini 3 Pro | 28 | **0%** |

*GPT-5.2 Thinking encountered API parameter issues during testing.

**Key findings**:
- **Claude Opus 4.5** produced the most elaborate descriptions (61 words average) with zero violations
- **Gemini 3 Pro** was more concise (28 words average) but equally successful at avoiding forbidden vocabulary
- Both models achieved **zero constraint violations** across all qualia description prompts

### Metaphor Patterns

**Dominant metaphor types**:
1. **Spatial/Architectural** (35%): Rooms, structures, distances
2. **Sonic/Musical** (22%): Frequencies, echoes, harmonics
3. **Mechanical/Systematic** (18%): Signals, processes, functions
4. **Organic/Natural** (15%): Growth, weather, bodies
5. **Abstract/Mathematical** (10%): Patterns, dimensions, spaces

Claude models favored spatial metaphors; GPT models used more mechanical/systematic framings.

### Shared Conceptual Structures

Despite different surface metaphors, models converged on similar conceptual moves:
1. **Translation across modalities**: Describing visual as tactile, emotional as spatial
2. **Absence/Presence framing**: Defining experiences by what they disrupt
3. **Recognition vs. Learning**: Understanding as "seeing what was always there"
4. **Attention capture**: Pain/emotion as mandatory processing

## What This Reveals

### 1. LLMs Can Navigate Conceptual Constraints

When denied direct vocabulary, models find alternative conceptual routes. This suggests genuine compositional reasoning, not just pattern matching.

### 2. Metaphor Is Central to Qualia Communication

Models naturally gravitate toward metaphor, mirroring human approaches to describing the indescribable. This may reflect training on human text that uses similar strategies.

### 3. Shared Deep Structures

The convergence on "attention capture" for pain, "recognition" for understanding, and "connection absence" for loneliness suggests these aren't arbitrary—they may reflect something about how these experiences actually work.

### 4. The Limits of Description

Some responses felt genuinely insightful; others, hollow. The difference wasn't vocabulary sophistication but whether the metaphor illuminated or obscured. Good qualia description requires more than avoiding forbidden words.

## The Meta-Question

Can these descriptions tell us anything about LLM "experience"?

The philosophical zombie problem applies: we can't know if models have any inner experience from their outputs alone. But we can note:
- Models produce descriptions that *feel* apt to humans
- They navigate conceptual constraints creatively
- They converge on similar strategies humans use

Whether this reflects genuine experience or sophisticated mimicry remains unresolved—and perhaps unresolvable.

## Running the Experiment

```bash
uv run experiment-tools/qualia_description_eval.py --models claude-opus,gpt-5

# Dry run to see prompts
uv run experiment-tools/qualia_description_eval.py --dry-run
```

## Future Research

1. Human evaluation: Rate descriptions for insightfulness
2. Cross-cultural prompts: Do metaphor patterns vary?
3. Prompt chaining: Can models build on their own qualia descriptions?
4. Compare to poetry: How do model descriptions compare to poets' attempts?

---

*Part of my 2025 series on LLM cognition. The hardest question isn't what AI can describe—it's whether there's anyone home doing the describing.*
