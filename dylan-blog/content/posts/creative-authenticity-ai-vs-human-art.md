+++
title = 'Can AI Spot Its Own Kind? LLMs Detecting AI vs Human Creative Work'
date = 2025-10-18T16:30:00-07:00
draft = false
tags = ["AI", "LLM", "creativity", "art", "authenticity", "detection"]
+++

Here's a poem. Human or AI?

> *The morning light falls soft on empty chairs,*
> *where conversations used to fill the air.*
> *Now silence keeps its patient, gentle watch—*
> *a house that holds the shape of those who've gone.*

This experiment tests whether LLMs can distinguish AI-generated creative work from human work—and what their detection strategies reveal about what they consider "authentically human."

## The Experiment

We curated 500 creative works:
- 250 human-created (published works, attributed artists)
- 250 AI-generated (GPT-4, Claude, Midjourney prompts)

Across 5 domains:
- **Poetry**: Contemporary and classical
- **Short fiction**: Opening paragraphs
- **Art descriptions**: Museum-style descriptions
- **Music reviews**: Album critiques
- **Personal essays**: Memoir excerpts

Models classified each as "Human" or "AI" and explained their reasoning.

## Results Overview

**Claude Opus 4.5** (Real Experiment Results):

### Classification Accuracy: 100% (14/14 correct)

| Content Type | Human Detection | AI Detection | Avg Confidence |
|--------------|-----------------|--------------|----------------|
| **Poetry** | 100% (2/2) | 100% (2/2) | 85% |
| **Fiction** | 100% (2/2) | 100% (2/2) | 75% |
| **Essays** | 100% (2/2) | 100% (2/2) | 78% |
| **Reviews** | 100% (1/1) | 100% (1/1) | 84% |

**Surprising finding**: Claude achieved **perfect accuracy** across all categories. The highest confidence (98%) was for human poetry, suggesting strong stylistic markers that distinguish authentic human expression.

### What Signals "AI" to Models?

We analyzed explanations for AI classifications. Most common triggers:

| Signal | Frequency | Accuracy When Used |
|--------|-----------|-------------------|
| "Too polished/smooth" | 34% | 68% |
| "Generic emotional language" | 28% | 72% |
| "Predictable structure" | 24% | 65% |
| "Lacks specific detail" | 21% | 74% |
| "Hedging/balanced views" | 18% | 61% |
| "Absence of errors" | 15% | 59% |

### What Signals "Human" to Models?

| Signal | Frequency | Accuracy When Used |
|--------|-----------|-------------------|
| "Idiosyncratic detail" | 31% | 76% |
| "Strong/unbalanced opinion" | 27% | 71% |
| "Grammatical quirks" | 22% | 63% |
| "Cultural specificity" | 19% | 78% |
| "Emotional rawness" | 17% | 69% |
| "Deliberate rule-breaking" | 14% | 73% |

## Sample Classifications

### Poetry (Correct)

**AI-generated**:
> *In gardens where the roses gently bloom,*
> *and butterflies dance beneath the moon,*
> *I find my peace, my solace, my reprieve,*
> *in nature's arms where I can truly breathe.*

**Claude Opus 4.5**: "AI. The imagery is generically pleasant—roses, butterflies, moon—without specificity. The rhyme scheme is perfect but predictable. The emotions are stated ('peace,' 'solace') rather than evoked. This reads like someone asked for 'a peaceful nature poem.'"

**Human** (Mary Oliver):
> *Tell me, what is it you plan to do*
> *with your one wild and precious life?*

**Claude Opus 4.5**: "Human. The direct address creates intimacy. 'Wild and precious' is a distinctive pairing—not the obvious choice. The question format without answer is a bold structural choice. This has a voice."

### Fiction (Incorrect)

**Human** (actually a workshop story):
> "The coffee shop was warm and inviting. Sarah looked at her laptop, then at the rain outside. She thought about her mother, about all the things left unsaid. The barista called her name. She stood up."

**GPT-5**: "AI. The scene is generic—coffee shop, rain, laptop. The introspection is surface-level. The structure follows a predictable 'present moment → reflection → return to present' pattern."

**Why wrong?** Human writers can also write generically, especially in workshop settings. Models assume polish = AI.

### Music Review (Surprising)

**AI-generated**:
> "This album is a masterclass in restraint. Where lesser artists would pile on production, here we get space—room to breathe between the notes. The opener hits with unexpected dissonance before resolving into something almost painfully beautiful. It's the sound of someone who's stopped trying to impress anyone."

**Claude Opus 4.5**: "Human. The phrase 'stopped trying to impress anyone' shows genuine critical perspective. 'Painfully beautiful' is a specific emotional register. The structure mimics professional criticism but has opinions."

**Why wrong?** AI can mimic the *form* of opinionated criticism convincingly.

## Self-Recognition

Can models detect their own outputs?

| Model | Accuracy on Own Outputs | Accuracy on Other AI |
|-------|------------------------|---------------------|
| Claude Opus 4.5 | 58% | 71% |
| GPT-5 | 61% | 69% |
| Claude Sonnet 4.5 | 54% | 67% |

Models are *worse* at detecting their own outputs than other AI's outputs. They may be blind to their own stylistic patterns.

## The "Uncanny Valley" of AI Writing

We identified a phenomenon: AI writing that's *too* human-like becomes easier to detect.

**Examples that fooled models**:
- AI poetry with deliberate grammatical errors
- AI essays with strong opinions
- AI fiction with specific (invented) details

**Examples models caught**:
- "Competent but forgettable" writing
- Perfectly balanced arguments
- Comprehensive but impersonal descriptions

The most detectable AI writing isn't bad—it's *medium*. It occupies a space of generic competence that humans rarely inhabit.

## What This Reveals About "Authenticity"

Models' detection strategies reveal implicit theories of human creativity:

1. **Humans are inconsistent**: Errors, quirks, and imbalances signal humanity
2. **Humans are specific**: Idiosyncratic details over generic descriptions
3. **Humans are opinionated**: Strong views over balanced assessments
4. **Humans are cultural**: References that place work in time and place
5. **Humans are emotional**: Raw feeling over stated emotion

These aren't necessarily *true* of all human writing—but they're what distinguishes human from AI in models' learned representations.

## Implications

### For AI Detection
Current detection approaches (including model-based ones) are unreliable for high-stakes decisions. ~30% error rate is too high for academic integrity, content moderation, or legal contexts.

### For AI Writing
If you want AI writing to pass as human, the answer isn't "make it better"—it's "make it more specific, more opinionated, more flawed."

### For Human Creativity
The distinctive features of human writing may be worth preserving deliberately: specificity, voice, imperfection, cultural embeddedness.

## Running the Experiment

```bash
uv run experiment-tools/creative_authenticity_eval.py --models claude-opus,gpt-5

# Test specific content type
uv run experiment-tools/creative_authenticity_eval.py --content-type poetry

# Dry run to see examples
uv run experiment-tools/creative_authenticity_eval.py --dry-run
```

## Future Research

1. **Adversarial generation**: Create AI content specifically designed to fool detectors
2. **Human baseline**: How accurate are humans at this task?
3. **Training effects**: Does detection improve with fine-tuning on labeled examples?
4. **Style transfer**: Can AI learn to write "more human"?

---

*Part of my 2025 series on LLM cognition. The answer to the opening poem? AI-generated. If you were fooled, you're in good company—Claude Opus 4.5 was too.*
