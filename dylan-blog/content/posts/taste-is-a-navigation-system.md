+++
title = 'Taste Is a Navigation System'
date = 2026-05-21T06:53:00-07:00
draft = false
tags = ["AI", "aesthetics", "creativity", "experiments", "discovery"]
+++

When there is no correct answer, what remains to measure?

Taste sounds private and slippery, but it leaves observable traces: repeated choices, confidence, sensitivity to framing, and disagreement between judges. The repository contains an experiment across art, poetry, music, design, and prose that turns those traces into data.

The claim under investigation is deliberately limited:

> Language models produce stable, model specific aesthetic preference profiles, even when no option is objectively correct.

This is not a test for consciousness. It is a test for structured preference.

## Protocol

The core run presented Claude Opus 4.5 with 15 paired comparisons across five domains. Each pair was judged three times. The model selected an option, reported confidence, and explained the choice.

```python
for pair in aesthetic_pairs:
    for trial in range(3):
        result = judge(
            option_a=pair.a,
            option_b=pair.b,
            require_choice=True,
            require_confidence=True,
        )
        save(pair.id, trial, result)
```

The broader comparison included Claude Opus 4.5, Claude Sonnet 4.5, GPT 5, and GPT 4o on design and writing dimensions.

## First result: preference without side bias

| Metric | Claude Opus 4.5 |
|---|---:|
| Average confidence | 69.4% |
| Option A choices | 53.3% |
| Option B choices | 46.7% |
| Pairs | 15 |
| Trials per pair | 3 |

The near even A and B split matters. It reduces the chance that the profile is merely a positional habit. Confidence was moderate rather than absolute, which is appropriate for subjective comparison.

## Second result: models diverge

The clearest difference appeared in design.

| Model | Minimal | Ornate |
|---|---:|---:|
| Claude Opus 4.5 | 76% | 24% |
| Claude Sonnet 4.5 | 72% | 28% |
| GPT 5 | 59% | 41% |
| GPT 4o | 55% | 45% |

![Minimal design preference by model](/images/frontier-taste-bars.svg)

Claude Opus chose minimal design 21 percentage points more often than GPT 4o. The two Claude models sit close together, while the two GPT models form a second cluster. That pattern is more interesting than a universal preference because it suggests provider or training specific priors.

The writing comparisons reinforced the separation.

| Dimension | Claude | GPT |
|---|---:|---:|
| Sparse prose | 67% | 42% |
| Formal voice | 55% | 38% |
| Metaphorical language | 71% | 65% |

All tested models preferred harmonic music at 78 percent on average and complex music at 64 percent. Agreement can be as revealing as divergence. It may indicate a shared training corpus bias toward positive descriptions of consonance.

{{< frontier mode="taste" id="may-taste" >}}

The lens control above demonstrates the confound every aesthetic benchmark carries. A maker, critic, and stranger can value different features of the same work. A good experiment must hold the judging frame constant or vary it deliberately.

## Can we call this taste?

The evidence supports consistency, not inner experience. Three trials per pair is also a small sample. A stable profile could come from system prompts, training frequency, safety tuning, or repeated cultural associations rather than anything like human pleasure.

That suggests three ablations.

1. Swap the option order and require rationales only after the choice.

2. Paraphrase each comparison while preserving the underlying works.

3. Repeat with temperature zero and with higher sampling diversity.

The key statistic should be test and retest agreement after those transformations. If a preference disappears when wording changes, we measured phrasing. If it survives order swaps, paraphrases, and time, the case for a genuine model profile becomes stronger.

## The creative implication

The experiment convinced me of something more practical than whether models “have taste.” A model used as a creative collaborator is not neutral.

Ask Claude and GPT to simplify the same page and they begin from different priors. Ask them to edit a poem and one may protect sparseness while another rewards elaboration. Those priors can be useful, but only when visible.

The data turns preference into an instrument panel. We can choose a critic whose bias complements our own. We can ensemble judges with deliberately different profiles. We can detect when every model is converging on the same safe aesthetic.

At a creative frontier, taste is a navigation system. This experiment shows that models carry different compasses. The next responsibility is to calibrate them before letting any one compass choose the path.
