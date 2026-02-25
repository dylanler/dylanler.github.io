+++
title = 'What I Learned Running a Long Horizon Memory Experiment on 4 A100 GPUs'
date = 2026-02-25T15:08:00-08:00
draft = false
tags = ["AI", "LLM", "memory", "experiments", "learning", "continuous-learning"]
+++

I wanted to answer one practical question.

Can a model keep learning over long sessions without slowly losing grip on earlier facts?

This post is a learning oriented walkthrough of one real campaign I ran. It focuses on understanding and decision making, not just reporting scores.

Code and implementation are here:

1. [GitHub repo: rlm-experiment-codex](https://github.com/dylanler/rlm-experiment-codex)
2. [Live report dashboard](http://10.1.7.101:8765/)

## Overview

I compared two memory methods with the same base model and the same datasets.

1. Text Buffer
2. Latent Pager Memory, or LPM

Setup:

1. Model: Qwen3 1.7B
2. Hardware: 4x A100 80GB
3. Data: oolong real and oolong synth
4. Focus: continuous learning behavior, context rot resistance, long horizon stability

I used three benchmark families.

1. Sentinel checks for baseline quality and contradiction
2. Context rot stress for distractor pressure and horizon growth
3. Continual stream tests for memory performance across long event sequences

## Architecture in plain language

The pipeline is simple to explain.

1. Build prepared examples from both datasets.
2. Run the same model with each memory method.
3. Score each run with the same metric stack.
4. Save per example records and summary files.
5. Render a dashboard and a static report.

The most important operational lesson came late. Top level parallelization was good at first, but heavy tail jobs created idle GPUs at the end. I fixed that by sharding the remaining hard context rot conditions across all four GPUs.

## Results

### Sentinel baseline

![Sentinel overview](/images/rlm-sentinel-overview.png)

![Sentinel quality vs latency](/images/rlm-sentinel-quality-vs-latency.png)

| Dataset | Method | Task score | Contradiction | Hall total | Mean time s | Mean total tokens |
|---|---|---:|---:|---:|---:|---:|
| oolong real | LPM | 0.0669 | 0.0819 | 0.0028 | 1.945 | 654.6 |
| oolong real | Text Buffer | 0.0515 | 0.1040 | 0.0111 | 46.511 | 55337.1 |
| oolong synth | LPM | 0.2878 | 0.1753 | 0.3694 | 1.664 | 459.3 |
| oolong synth | Text Buffer | 0.2500 | 0.2055 | 0.3639 | 4.335 | 2469.8 |

What this taught me:

1. LPM improved task score in both datasets.
2. LPM reduced contradiction in both datasets.
3. Latency difference on oolong real was very large.

### Context rot behavior

![Context rot heatmap](/images/rlm-context-rot-heatmap.png)

![Context robustness curves](/images/rlm-context-robustness-curves.png)

How to read this section:

1. Distractor ratio increases retrieval pressure.
2. Horizon multiplier increases memory distance.
3. Falling task score in harder cells means context rot sensitivity.
4. Missing cells mean active jobs were still running at snapshot time.

What I learned from context rot so far:

1. Some LPM regions are stable even at higher stress.
2. Text Buffer has a heavier runtime tail in hard real dataset conditions.
3. Hard corner data should be interpreted only after all shards finish.

### Continual stream behavior

![Continual capacity](/images/rlm-continual-capacity.png)

In this event budget, capacity effects were smaller than I expected. That suggests the current stream was not yet harsh enough to separate methods strongly by memory size alone.

## Training setup and why it matters

This campaign did not update model weights.

1. Base weights stayed frozen.
2. Memory behavior was the thing under test.
3. Seed and split were fixed for comparability.

This matters for learning because it isolates memory system quality from optimizer noise.

## Ablations

The cleanest ablation was method swap with everything else fixed.

| Dataset | LPM minus Text Buffer task delta | LPM minus Text Buffer contradiction delta | LPM speedup |
|---|---:|---:|---:|
| oolong real | +0.0154 | -0.0221 | 23.92x |
| oolong synth | +0.0378 | -0.0302 | 2.60x |

This is the exact ablation I care about most in production planning because it combines quality and cost in the same direction.

## Visual timeline of the run

![Runtime timeline](/images/rlm-runtime-timeline.png)

Timeline insight:

1. Sentinel finished early.
2. Continual stream finished next.
3. Context rot hard shards dominated tail time.
4. Tail sharding recovered utilization.

## Hypotheses and what changed in my thinking

I started with three hypotheses.

1. LPM should improve cost while preserving or improving quality.
2. Context rot should worsen as distractors and horizon increase.
3. Continual memory should keep high hit rate over long sessions.

Current interpretation:

1. Hypothesis one looks strong.
2. Hypothesis two is supported in trend, with final hard corner pending.
3. Hypothesis three looks promising but needs longer stream stress to verify limit behavior.

## Examples from outputs

Real per example records matter because averages can hide failure modes.

1. Several count questions were answered exactly.
2. Some hard count queries still returned unknown.
3. Context classification could succeed on one item and fail on a nearby item in the same condition.

This pattern is a reminder that reliability is about tails, not only means.

## Future direction for RLM models

I want this section to be honest. The chart below is a forecast, not measured data.

![Forecast direction](/images/rlm-forecast-direction.png)

Forecast assumptions:

1. Better memory routing and retrieval gating continue to improve retention.
2. Contradiction controls become first class objectives in memory systems.
3. Cost efficiency improves as latent memory representations mature.

Prediction table:

| Year | What likely improves | What will still be hard |
|---|---|---|
| 2026 | Better memory routing, lower latency variance | Very long context factual consistency |
| 2027 | More stable contradiction control in long dialogs | Generalization under domain shift |
| 2028 | Memory systems become default in production agent stacks | Evaluation of rare tail failures at scale |

My current belief is that the winning RLM direction is not one trick. It is a blend of better memory representation, better retrieval policies, and better stress evaluation loops that run continuously.

## What to do next if you are building this yourself

1. Keep a sentinel suite running all the time.
2. Add explicit context rot stress tests early.
3. Track tail latency and tail correctness, not only mean values.
4. Keep your reporting pipeline automatic so you can learn every day, not at the end.

This project taught me that long horizon capability is not a single benchmark property. It is an engineering discipline across memory design, evaluation design, and runtime scheduling.
