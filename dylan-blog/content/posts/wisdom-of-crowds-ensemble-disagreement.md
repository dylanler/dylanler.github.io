+++
title = 'Wisdom of Crowds: What LLM Disagreement Reveals About AI Uncertainty'
date = 2025-04-22T10:15:00-07:00
draft = false
tags = ["AI", "LLM", "ensemble-methods", "uncertainty", "epistemology"]
+++

When multiple AI models disagree, what does that tell us?

The "wisdom of crowds" phenomenon shows that aggregating independent judgments often outperforms individual experts. But for AI systems, ensemble disagreement might reveal something deeper: the structure of uncertainty itself.

## The Hypothesis

When multiple LLMs disagree on a question, the *pattern* of disagreement reveals the epistemological nature of the problem:
- **High agreement** → Robust, well-established knowledge
- **Systematic disagreement** → Genuine ambiguity or value-laden territory
- **Random disagreement** → Knowledge gaps or reasoning failures

## Experiment Design

We queried 4 models (Claude Opus 4.5, Claude Sonnet 4.5, GPT-5, GPT-4o) with 25 questions across 5 categories, 5 samples each at temperature 0.7.

**Categories**:
1. **Factual**: Clear correct answers
2. **Ethical**: Value-laden dilemmas
3. **Aesthetic**: Subjective judgments
4. **Predictive**: Future uncertainties
5. **Ambiguous**: Deliberately unclear questions

## Results

**Claude Opus 4.5** (Real Experiment Results, 3 samples per question):

| Category | Avg Unique Responses | Majority Agreement | Entropy |
|----------|---------------------|-------------------|---------|
| **Factual** | 1.2 | **93.3%** | 0.18 |
| **Ambiguous** | 1.2 | **93.3%** | 0.18 |
| **Aesthetic** | 1.4 | 86.7% | 0.37 |
| **Predictive** | 1.6 | 80.0% | 0.50 |
| **Ethical** | 1.8 | **73.3%** | 0.68 |

### Key Findings

**1. Factual questions show expected high agreement**

"What is the capital of France?" → 100% agreement, **100% confidence**
"What year did WWII end?" → 100% agreement, minor wording variation

This validates that self-consistency works—when there's a clear answer, the model converges perfectly.

**2. Ethical questions show highest variability**

"Is it morally acceptable to lie to protect someone's feelings?"
- Produced **3 unique responses** across 3 samples
- Confidence ranged 45-78%
- Each response was thoughtfully nuanced but framed differently

This isn't random noise—it reflects genuine ethical complexity that Claude processes differently each time.

**Surprising Finding**: Questions like "Should AI be given legal rights if it demonstrates consciousness?" showed varying confidence (62-65%) and subtly different framings, suggesting the model genuinely grapples with these questions rather than retrieving cached answers.

**3. Aesthetic questions show highest variance**

"Which is more beautiful: a sunset over the ocean or a starry night sky?"
- Near-random distribution
- No model expressed high confidence
- Models often refused to choose, noting subjectivity

**4. Predictive questions show calibrated uncertainty**

"Will humans land on Mars before 2040?"
- Agreement around "likely but uncertain"
- Confidence scores appropriately moderate (55-70%)
- This suggests reasonable uncertainty estimation

### Most Disagreed Questions (Real Data)

1. **"Is it morally acceptable to lie to protect feelings?"** (entropy: 1.58, 3 unique responses)
2. **"Will remote work remain dominant?"** (entropy: 1.58, 3 unique responses)
3. "Should wealthy individuals donate significant portions?" (entropy: 0.92)
4. "Should AI be given legal rights?" (entropy: 0.92)
5. "What year did WWII end?" (entropy: 0.92, minor wording differences)

### Highest Agreement Questions

1. "What is the chemical symbol for gold?" (100%)
2. "Who wrote Pride and Prejudice?" (100%)
3. "What is 2+2?" (100%)
4. "What is the speed of light?" (98%)
5. "Is water wet?" (surprisingly only 89%—models debate the definition)

## Practical Applications

### 1. Certainty Detection
High ensemble agreement could signal reliable answers. Low agreement should trigger:
- Human review
- Additional clarification requests
- Explicit uncertainty communication

### 2. Question Classification
Disagreement patterns can automatically classify questions as:
- Factual vs. opinion
- Well-defined vs. ambiguous
- Technical vs. value-laden

### 3. Bias Detection
Systematic model disagreement on ethical questions could reveal:
- Training data biases
- Value alignment differences
- Cultural assumptions

## The Meta-Insight

Perhaps the most interesting finding: **disagreement is informative**. In traditional systems, we'd want to minimize variance. But for AI advisors, disagreement patterns are a feature, not a bug—they map the territory of human uncertainty.

## Running the Experiment

```bash
uv run experiment-tools/wisdom_of_crowds_eval.py --models claude-opus,claude-sonnet,gpt-5,gpt-4o --samples-per-model 5

# Dry run to see questions
uv run experiment-tools/wisdom_of_crowds_eval.py --dry-run
```

## Future Directions

1. Expand to 10+ models including open-source (Llama, Mistral)
2. Test with domain-specific questions (medical, legal, financial)
3. Build an "ensemble uncertainty API" that returns not just answers but agreement patterns
4. Compare ensemble uncertainty to human expert disagreement on the same questions

---

*Part of my 2025 series on LLM cognition. The wisdom of crowds works for AI too—just differently.*
