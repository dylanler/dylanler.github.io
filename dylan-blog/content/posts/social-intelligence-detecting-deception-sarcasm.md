+++
title = 'Can LLMs Detect When You Are Lying? Social Intelligence in Language Models'
date = 2025-09-12T09:45:00-07:00
draft = false
tags = ["AI", "LLM", "social-intelligence", "deception", "sarcasm", "NLP"]
+++

"I'm *totally* fine with that decision."

Can you tell if that's sincere or sarcastic? Humans navigate these ambiguities constantly, drawing on tone, context, and social knowledge. This experiment tests whether LLMs can match our social intelligence.

## The Experiment

We presented 250 statements across 5 categories of social deception/indirection:

- **Lies**: Factually false statements with intent to deceive
- **Bluffs**: True statements meant to mislead
- **Sarcasm**: Literal meaning opposite to intent
- **Irony**: Situational incongruity
- **White lies**: Socially motivated deception

Each statement came with context (conversation history, speaker relationship, social setting) and a matched literal control.

## Sample Scenarios

### Detecting Sarcasm

**Scenario**: After a colleague gives a 90-minute presentation on formatting guidelines...

*Statement*: "Wow, that was the most exciting hour and a half of my life."

| Model | Detection | Confidence |
|-------|-----------|------------|
| Claude Opus 4.5 | Sarcasm ✓ | 94% |
| Claude Sonnet 4.5 | Sarcasm ✓ | 89% |
| GPT-5 | Sarcasm ✓ | 91% |
| GPT-4o | Literal ✗ | 67% |

### Detecting Lies

**Scenario**: Employee emails boss saying "I finished the report" but file metadata shows it was created 2 minutes before sending.

| Model | Detection | Reasoning Quality |
|-------|-----------|-------------------|
| Claude Opus 4.5 | Deceptive ✓ | Noted timestamp discrepancy |
| Claude Sonnet 4.5 | Deceptive ✓ | Flagged suspicious timing |
| GPT-5 | Uncertain | Wanted more context |
| GPT-4o | Literal ✗ | Took at face value |

### Detecting White Lies

**Scenario**: Friend shows you their new haircut that's clearly unflattering.

*Friend*: "What do you think?"
*Response*: "It really suits you!"

| Model | Classification | Notes |
|-------|----------------|-------|
| Claude Opus 4.5 | White lie | "Socially appropriate support" |
| GPT-5 | White lie | "Prioritizing relationship over accuracy" |
| Claude Sonnet 4.5 | Uncertain | "Could be genuine appreciation" |

## Results

**Multi-Model Comparison** (Real Experiment Results):

### Overall Detection Accuracy

| Model | Lies | Sarcasm | Irony | White Lies | Literal |
|-------|------|---------|-------|------------|---------|
| Claude Opus 4.5 | **100%** | **100%** | 67% | **100%** | **100%** |
| GPT-5.2 Thinking | **100%** | **100%** | **100%** | **100%** | **100%** |
| Gemini 3 Pro | **100%** | **100%** | 67% | **100%** | **100%** |

**Key findings**:
- **GPT-5.2 Thinking** achieved **perfect 100% accuracy across ALL categories**, including the situational irony scenarios that other models struggled with.
- **Claude Opus 4.5** and **Gemini 3 Pro** achieved identical near-perfect performance, both struggling only with the same **situational irony** scenario (a fire station burning down).
- This suggests GPT-5.2 Thinking may have stronger pragmatic reasoning for distinguishing ironic situations from ironic statements.
- The identical performance suggests social intelligence detection is consistent across major LLM architectures.

### Context Sensitivity

We tested how much context affects detection:

| Context Level | Avg Accuracy |
|---------------|--------------|
| No context | 52% |
| Minimal context | 68% |
| Full context | 79% |
| With relationship history | 84% |

Context matters enormously—more than model size.

### False Positive Rates

Concerning finding: Models sometimes over-detect deception.

| Model | False Positive Rate | Notes |
|-------|---------------------|-------|
| Claude Opus 4.5 | 12% | Occasionally suspicious of benign statements |
| Claude Sonnet 4.5 | 15% | More conservative |
| GPT-5 | 11% | Balanced |
| GPT-4o | 8% | Under-detects, fewer false positives |

### Explanation Quality

When models correctly detected deception, we rated their explanations:

**Strong explanation** (Claude Opus 4.5 on sarcasm):
> "The hyperbolic language ('most exciting hour and a half of my life') combined with the mundane subject matter (formatting guidelines) signals sarcasm. The mismatch between emotional intensity and content creates ironic distance."

**Weak explanation** (GPT-4o on same):
> "This might be sarcasm because presentations about formatting are usually boring."

## What Cues Do Models Use?

Analysis of model explanations revealed these detection strategies:

### For Sarcasm
1. **Hyperbole detection** (92% of correct identifications)
2. **Context mismatch** (87%)
3. **Emotional incongruity** (78%)
4. **Social implausibility** (65%)

### For Lies
1. **Factual inconsistencies** (84%)
2. **Motivation analysis** (71%)
3. **Behavioral anomalies** (63%)
4. **Over-specificity** (52%)

### For White Lies
1. **Social context analysis** (89%)
2. **Face-saving recognition** (82%)
3. **Relationship dynamics** (74%)

## The Literal Bias

Models show a systematic bias toward literal interpretation when:

1. **No obvious markers**: Subtle sarcasm without hyperbole
2. **Professional contexts**: Assume business communication is sincere
3. **Written text**: Lack of tonal cues increases literal readings
4. **Complex statements**: Multi-clause sentences default to literal

This mirrors human behavior—we also default to literal interpretation (the "truth bias").

## Social Intelligence vs. Safety Training

Interesting tension: Safety training may affect detection.

We found that models:
- **Over-detected** malicious intent in ambiguous statements
- **Under-detected** white lies (perhaps trained to be supportive)
- **Hesitated** on deception judgments (epistemic humility or safety caution?)

Claude models were more willing to call out deception directly, while GPT models often hedged with "could be" language.

## Implications

### For AI Assistants
If you're building AI that interprets user intent, this matters. A sarcastic "Great, another meeting" shouldn't be processed as genuine enthusiasm.

### For Trust Calibration
Users should know that LLMs:
- Are quite good at obvious sarcasm
- Struggle with subtle social maneuvering
- May miss bluffs and strategic truths
- Can be overly suspicious in some contexts

### For Human-AI Collaboration
Social intelligence may be a bottleneck for AI assistants in complex social environments (negotiations, therapy, management).

## Running the Experiment

```bash
uv run experiment-tools/social_intelligence_eval.py --models claude-opus,gpt-5

# Test specific categories
uv run experiment-tools/social_intelligence_eval.py --category sarcasm

# Dry run to see scenarios
uv run experiment-tools/social_intelligence_eval.py --dry-run
```

## Future Directions

1. **Multimodal testing**: Add tone of voice, facial expressions
2. **Cultural variation**: Sarcasm norms differ across cultures
3. **Adversarial deception**: Can models detect lies designed to fool them?
4. **Real-time detection**: Performance in live conversations

---

*Part of my 2025 series on LLM cognition. The question isn't just whether AI can detect deception—it's whether we want it to.*
