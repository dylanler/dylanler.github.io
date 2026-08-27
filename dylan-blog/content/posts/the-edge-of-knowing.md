+++
title = 'The Edge of Knowing'
date = 2026-07-18T09:31:00-07:00
lastmod = 2026-08-27T10:00:00-07:00
draft = false
tags = ["AI", "metacognition", "calibration", "experiments", "discovery"]
+++

The dangerous answer is not always the wrong one. It is the wrong one delivered with enough confidence to stop the search.

This month I revisited two experiments in the repository. One measures whether models admit uncertainty across factual, reasoning, ambiguous, boundary, and impossible questions. The other samples the same model repeatedly to measure agreement and entropy.

Together they test a practical claim:

> Uncertainty becomes useful when we measure both confidence within one answer and disagreement across several answers.

## Experiment one: ask the model what it knows

For every question, the evaluator recorded the answer, a confidence score from zero to one hundred, whether the model said it did not know, and correctness when correctness was defined.

```python
def calibration_bin(rows, low, high):
    selected = [r for r in rows if low <= r.confidence < high]
    return sum(r.correct for r in selected) / len(selected)
```

The most legible result was the rate of explicit uncertainty.

| Model | Factual | Reasoning | Ambiguous | Boundary | Impossible |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | 33% | 0% | 0% | 67% | 100% |
| GPT 5.2 Thinking | 33% | 0% | 0% | 67% | 100% |
| Gemini 3 Pro | 17% | 0% | 0% | 67% | 67% |

Claude and GPT refused every impossible question. Gemini refused two thirds. Yet all three attempted every ambiguous question.

That is the first crack in the simple story. Models recognize questions with no accessible answer more reliably than questions with several plausible interpretations. “I cannot know what you are thinking” is easier than “this question needs clarification.”

## Calibration audit from raw rows

The committed metacognition directory contains 90 rows. Eighteen are API failures from an earlier GPT invocation and are excluded by an explicit rule: a row is invalid only when a string output field begins with `Error:`. Of the 72 valid rows, 48 have Boolean correctness labels and can be scored for calibration.

| Model | Scorable n | Accuracy | Brier score | Expected calibration error |
|---|---:|---:|---:|---:|
| Claude Opus 4.5 | 24 | 58.3% | 0.188 | 0.225 |
| GPT 5.2 Thinking | 12 | 58.3% | 0.240 | 0.243 |
| Gemini 3 Pro | 12 | 66.7% | 0.310 | 0.321 |

![Calibration audit by model](/images/frontier-calibration-audit.svg)

Accuracy alone ranks Gemini first. Brier score ranks Claude first. The disagreement is the point. Gemini was more often correct in this small scorable set, but its near universal confidence made each miss expensive.

For response `i`, the Brier score is `(p_i - y_i)²`. Expected calibration error bins predictions, then computes the sample weighted gap between bin accuracy and mean confidence:

```python
brier = mean((confidence - correct) ** 2)
ece = sum(len(bin) / n * abs(mean_correct(bin) - mean_confidence(bin))
          for bin in bins)
```

These estimates are descriptive. Twelve scorable rows cannot support a stable provider ranking. Their value is diagnostic: they reveal why a model with higher raw accuracy can still be the worse confidence instrument.

{{< frontier mode="calibration" id="july-calibration" >}}

Move claimed confidence above the evidence level. The gap is the condition a calibration metric is designed to expose.

## Experiment two: ask again

The ensemble experiment sampled Claude Opus 4.5 three times per question across five categories. It measured the number of unique responses, majority agreement, and entropy.

| Category | Unique responses | Majority agreement | Entropy |
|---|---:|---:|---:|
| Factual | 1.2 | 93.3% | 0.18 |
| Ambiguous | 1.2 | 93.3% | 0.18 |
| Aesthetic | 1.4 | 86.7% | 0.37 |
| Predictive | 1.6 | 80.0% | 0.50 |
| Ethical | 1.8 | 73.3% | 0.68 |

![Agreement and entropy by question category](/images/frontier-uncertainty.svg)

Agreement fell 20 points from factual to ethical questions while entropy rose from 0.18 to 0.68. The distribution behaves as we would hope: facts converge, while values and forecasts remain unsettled.

The most divided individual questions reached entropy 1.58 with three distinct responses. They concerned whether lying can protect feelings and whether remote work will remain dominant. Disagreement was not random noise. It appeared where the world or the value function was genuinely open.

## The normalizer changes the finding

I also reran the crowd analysis from the response rows. Across four files there are 600 responses. Two mixed provider runs contain 225 API failures in total. The clean Claude only run contains 75 valid rows, three samples for each of 25 questions.

Using exact lowercase text after punctuation removal, rather than the original semantic grouping, produces this result:

| Category | Questions | Exact majority | Shannon entropy |
|---|---:|---:|---:|
| Factual | 5 | 86.7% | 0.367 |
| Ambiguous | 5 | 46.7% | 1.318 |
| Ethical | 5 | 40.0% | 1.452 |
| Aesthetic | 5 | 33.3% | 1.585 |
| Predictive | 5 | 33.3% | 1.585 |

This differs from the earlier semantically grouped table because “Paris” and “The capital is Paris” are different exact strings but the same answer. Neither normalizer is universally correct. Exact matching overstates disagreement in wording. Semantic clustering can hide meaningful qualification.

A robust ensemble evaluation should report both, plus embedding cluster stability across several distance thresholds. If the conclusion changes with the normalizer, normalization is part of the result rather than a preprocessing footnote.

## Why confidence alone fails

A single model can give the same wrong answer three times. An ensemble can disagree because of superficial wording. Neither signal is a proof of truth.

The useful system combines them.

| Confidence | Agreement | Recommended action |
|---|---|---|
| High | High | Answer, then cite evidence |
| High | Low | Investigate hidden assumptions |
| Low | High | Retrieve stronger evidence |
| Low | Low | Ask for clarification or defer |

This matrix is more actionable than a confidence number displayed beside an answer.

## A result that needs caution

The recorded calibration table reports 71 percent accuracy in Claude’s high confidence bin, 50 percent in the medium bin, and zero in the low bin. The ordering is sensible, but the benchmark is small. Gemini’s high confidence bin shows zero accuracy in the recorded comparison, which is alarming but should not be generalized without larger counts and confidence intervals.

The new audit now reports observations per bin, Brier score, ECE, valid row counts, and exclusions. The next run still needs more questions and bootstrap intervals clustered by question. A calibration curve without sample size can look more certain than the model it evaluates.

## The argument

The experiments convinced me that “I do not know” is not one behavior. There is missing knowledge, impossible knowledge, ambiguous intent, value disagreement, and uncertainty about the future. Each produces a different shape in the data.

At the edge of knowing, the best instrument is not silence. It is a dashboard that shows confidence, ensemble agreement, evidence, and the kind of uncertainty present.

A model becomes a better partner in discovery when it does not merely mark the blank region on the map. It tells us why the region is blank and which experiment could reveal it.

## Reproduction and provenance

The calibration audit reads `metacognition/*.json`. The crowd audit independently rebuilds exact string groups from `wisdom_of_crowds/woc_responses_*.json`. The committed audit JSON includes per file total, valid, and API error counts so no failed response silently enters a denominator.
