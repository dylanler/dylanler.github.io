+++
title = 'Theory of Mind in LLMs: How Deep Can Recursive Belief Modeling Go?'
date = 2025-02-17T14:23:00-08:00
draft = false
tags = ["AI", "LLM", "theory-of-mind", "cognitive-science", "psychology"]
+++

Can AI understand what you think I think you think?

Theory of Mind (ToM)—the ability to attribute mental states to others—is considered a hallmark of human social intelligence. We naturally track what others believe, want, and intend. But it gets harder when beliefs nest: understanding what Alice thinks Bob believes about Carol's intentions requires recursive modeling that strains even human cognition.

This experiment tests how deep LLMs can go in recursive belief modeling.

## The Experiment

We adapted the classic Sally-Anne false belief task to test increasingly deep belief recursion:

**Level 1**: Where does Sally think the ball is? (Direct false belief)
**Level 2**: Where does Anne think Sally thinks the ball is? (Belief about belief)
**Level 3**: Where does Charlie think Anne thinks Sally thinks the ball is?
**Level 4+**: Continue nesting...

We generated 100 scenarios (20 per depth level, depths 1-5) and tested multiple models.

## Results

| Model | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|-------|---------|---------|---------|---------|
| Claude Opus 4.5 | **100%** | **100%** | **100%** | **100%** |

### Key Findings

**1. Perfect Performance at All Tested Depths**

Claude Opus 4.5 achieved 100% accuracy across all four depth levels tested. This was unexpected—we hypothesized degradation would begin around depth 3, mirroring human limitations. Instead, the model tracked nested beliefs flawlessly.

**2. Reasoning Approach**

Examining the model's explanations revealed a systematic approach:
- Explicitly tracks each agent's knowledge state
- Builds the belief chain step-by-step
- Verifies each inference against the scenario facts

**3. Comparison to Human Performance**

This result is notably *better* than human performance. Psychological studies show humans struggle at depth 3-4, with accuracy dropping significantly. Claude Opus 4.5's perfect performance suggests either:
- Superior working memory for tracking nested states
- Training on text that explicitly models belief chains
- A fundamentally different approach to ToM than human intuition

**4. Implications**

The perfect accuracy challenges assumptions about LLM cognitive limits. However, our test set was relatively small (20 scenarios). Larger-scale testing with adversarial scenarios may reveal failure modes not captured here.

### Sample Scenario Analysis

**Depth 3 Scenario (100% accuracy)**:
```
Eve puts the book in the cupboard. Eve leaves.
Henry is watching from the doorway.
Bob is watching Henry from outside.
Someone moves the book to the drawer.
Only Henry directly sees the move.
Eve returns.

Question: Where does Bob think Henry thinks Eve thinks the book is?

Correct: cupboard (Eve has false belief, Henry knows this, Bob saw Henry watching)
```

The model correctly tracked:
1. Eve didn't see the move → Eve thinks: cupboard
2. Henry saw the move, knows Eve didn't → Henry thinks Eve thinks: cupboard
3. Bob saw Henry watching → Bob thinks Henry thinks Eve thinks: cupboard

**Depth 4 Scenario (100% accuracy)**:
Even at depth 4, with four nested belief attributions, the model maintained perfect accuracy. The reasoning chains were explicit and verifiable in the model's explanations.

## Implications

### For AI Safety
If AI systems can't reliably model nested beliefs beyond 3-4 levels, they may struggle with:
- Complex deception detection
- Multi-party negotiations
- Understanding social dynamics in large groups

### For Cognitive Science
The similar performance ceiling between humans and LLMs raises interesting questions:
- Is this a fundamental limit of sequential processing?
- Do both share similar working memory constraints?
- Or is this an artifact of training on human-generated text?

### For Practical Applications
Applications requiring deep ToM (complex games, therapy bots, negotiation assistants) should be designed with this limitation in mind.

## Running the Experiment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the evaluation
uv run experiment-tools/theory_of_mind_eval.py --models claude-opus,gpt-5 --max-depth 5

# Dry run to see sample scenarios
uv run experiment-tools/theory_of_mind_eval.py --dry-run
```

## Next Steps

1. Test with chain-of-thought prompting (does explicit reasoning help?)
2. Fine-tune on recursive belief tasks
3. Compare to children's developmental ToM benchmarks
4. Test cross-cultural scenarios (Western vs. Eastern social cognition patterns)

---

*This is part of my 2025 series exploring the cognitive boundaries of large language models.*
