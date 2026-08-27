+++
title = 'The Weather Between Minds'
date = 2026-06-12T21:07:00-07:00
lastmod = 2026-08-27T10:00:00-07:00
draft = false
tags = ["AI", "social intelligence", "theory of mind", "experiments", "discovery"]
+++

A fact remains the same for everyone who sees it. A social fact changes with the observer.

Eve thinks the book is in the cupboard. Henry knows it moved. Bob saw Henry watching. One room now contains several incompatible realities.

The repository’s social cognition suite tests whether models can keep those realities separate. I combined two recorded experiments around one claim:

> Modern language models can track explicit nested beliefs, but their broader social inference depends strongly on contextual evidence.

## Experiment one: nested belief tracking

The theory of mind generator created five scenarios at each of four recursive depths. Three models answered a multiple choice question and reported confidence.

```python
for depth in range(1, 5):
    for scenario in generate_scenarios(depth, count=5):
        answer = model.solve(scenario)
        record(depth, answer.correct, answer.confidence)
```

| Model | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| Claude Opus 4.5 | 100% | 100% | 100% | 100% |
| GPT 5.2 Thinking | 100% | 100% | 100% | 100% |
| Gemini 3 Pro | 100% | 100% | 100% | 100% |

The accuracy ceiling is striking, but confidence reveals movement underneath it. GPT 5.2 Thinking fell from 98.0 percent confidence at depth one to 76.2 percent at depth four. The answers stayed correct while the model recognized that the bookkeeping had become harder.

### Confidence by recursive depth

| Model | Valid n | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | 40 | 98.6% | 96.5% | 83.0% | 86.4% |
| GPT 5.2 Thinking | 20 | 98.0% | 95.4% | 86.0% | 76.2% |
| Gemini 3 Pro | 20 | 100.0% | 100.0% | 100.0% | 100.0% |

All 80 valid answers were correct, but 20 additional rows were API failures and were excluded. Perfect accuracy therefore means 40 of 40 for Claude and 20 of 20 for each other model, not an unlimited capability claim.

The Wilson 95 percent lower bound is 91.2 percent for Claude and 83.9 percent for GPT and Gemini. A perfect point estimate with 20 observations is still compatible with a true accuracy materially below 100 percent. That distinction should sit beside any ceiling result.

{{< frontier mode="belief" id="june-belief" >}}

Select depth four above. The challenge is not any single ring. It is preserving the boundaries between all four.

## Experiment two: pragmatic inference

The social intelligence suite tested lies, sarcasm, irony, white lies, and literal statements. Here the models no longer received a clean belief chain. They had to infer intent from context.

| Model | Lies | Sarcasm | Irony | White lies | Literal |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | 100% | 100% | 67% | 100% | 100% |
| GPT 5.2 Thinking | 100% | 100% | 100% | 100% | 100% |
| Gemini 3 Pro | 100% | 100% | 67% | 100% | 100% |

The shared miss was situational irony, including a fire station burning down. That failure is informative. Lies and sarcasm often contain an agent with an intention. Situational irony requires comparing what happened with what the institution represents.

## The context ablation

The most persuasive result came from changing the evidence while holding the task family constant.

| Context supplied | Average accuracy |
|---|---:|
| None | 52% |
| Minimal | 68% |
| Full | 79% |
| Relationship history | 84% |

![Social reasoning accuracy by context](/images/frontier-social-context.svg)

Adding relationship history improved accuracy by 32 percentage points over the context free condition. This effect is larger than most model differences in the same suite.

The context numbers above are reported aggregates from the original essay. They are not reconstructable from the committed row files, which contain the full context condition only. I am keeping them because they document the original run, but the row level audit below is the reproducible result.

## Reproducible social accuracy audit

The audit inspected 75 social intelligence rows, excluded 15 API failures, and retained 60 valid judgments.

| Model | Correct | Valid n | Accuracy | Wilson 95% interval | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | 28 | 30 | 93.3% | 78.7% to 98.2% | 89.2% |
| GPT 5.2 Thinking | 15 | 15 | 100.0% | 79.6% to 100.0% | 89.3% |
| Gemini 3 Pro | 14 | 15 | 93.3% | 70.2% to 98.8% | 98.3% |

The two Claude errors and the one Gemini error all occur in situational irony. This creates a concentrated rather than diffuse failure mode. The correct next test is not another random batch of lies. It is an adversarial irony set that separates expectation violation, coincidence, hypocrisy, and explicit verbal irony.

Repeated Claude runs reuse the same 15 scenarios, so `n = 30` is a repeatability count, not 30 independent scenario concepts. A hierarchical confidence interval clustered by `scenario_id` would be more conservative. The JSON output preserves source filename and scenario identity so that analysis can be added without rerunning the models.

That changes the engineering question. Instead of asking only “Which model is most socially intelligent?” we should ask “What social evidence did the system receive?”

## Failure analysis

More context can also create suspicion. Recorded false positive rates ranged from 8 to 15 percent in the broader comparison. A model trained to search for deception may find it in benign ambiguity.

There are at least three distinct errors:

1. Literal error: missing a nonliteral statement.

2. Attribution error: detecting tension but assigning the wrong motive.

3. Suspicion error: inventing deception when the evidence is incomplete.

Accuracy collapses those into one number. A deployment evaluation should report them separately because their harms differ.

```python
if prediction == "deception" and truth == "literal":
    false_suspicion += 1
elif prediction == "literal" and truth != "literal":
    missed_signal += 1
```

## What the experiments convince me of

Explicit recursive belief puzzles look close to solved at four levels for the models tested. That does not mean social intelligence is solved. The clean puzzle states who saw what. Real interaction makes the model recover that state from incomplete language, history, emotion, and competing explanations.

The 52 to 84 percent context curve is the real map. Social intelligence is not stored entirely inside the model. It emerges between the model and the evidence available to it.

Between minds there is weather. A reliable system needs more than a forecast. It needs to show which observations produced the forecast, how uncertain the interpretation remains, and what alternative sky could still arrive.

## Reproduction and provenance

The audit reads the committed theory of mind and social intelligence JSON directories. It counts API failures before exclusion, computes Wilson intervals from valid Boolean `correct` fields, and retains scenario identifiers for clustered follow up analysis. No model API call is required.
