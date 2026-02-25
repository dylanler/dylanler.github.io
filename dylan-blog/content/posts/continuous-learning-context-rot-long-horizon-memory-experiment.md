+++
title = 'What I Learned Running a Long Horizon Memory Experiment on 4 A100 GPUs'
date = 2026-02-25T14:50:00-08:00
draft = false
tags = ["AI", "LLM", "memory", "experiments", "learning"]
+++

I wanted to answer one practical question.

Can a model keep learning over long sessions without slowly losing grip on earlier facts?

That sounds simple, but it is the exact failure mode that appears in real assistants. A model starts well, then quality drifts, contradictions grow, and old context gets buried.

So I ran a full memory stress campaign on 4 A100 80GB GPUs, comparing two approaches.

1. Text Buffer
2. Latent Pager Memory, or LPM

This post is not meant to be a formal paper. It is a learning note that explains what I built, what I measured, what surprised me, and what I would do next.

## Overview

I used one model family, Qwen3 1.7B, and two datasets, oolong real and oolong synth.

Then I tested memory quality with three lenses.

1. Sentinel evaluation, to check baseline quality, contradiction, and hallucination.
2. Context rot stress, to inject distractors and extend horizon length.
3. Continual long horizon stream, to see whether memory remains usable across event sequences.

The point was not to chase one benchmark score. The point was to understand memory behavior under pressure.

## Architecture in Plain Language

The pipeline has a clear flow.

1. Prepare examples from oolong real and oolong synth.
2. Run the same base model with two memory methods.
3. Evaluate each method across the three benchmark families.
4. Save per example outputs, summaries, logs, and live dashboards.
5. Build a static report page with charts.

One operational lesson mattered a lot. Top level parallelization was not enough late in the run. After most jobs finished, one heavy context rot job held the wall clock hostage. I had to split the tail into four GPU pinned shards to use all hardware.

## Results

### Sentinel results

This chart compares baseline task quality, contradiction, and latency profile.

![Sentinel results](/images/rlm-sentinel-overview.png)

What I learned from sentinel:

1. LPM was faster on both datasets.
2. On this snapshot, LPM also had better task score and lower contradiction.
3. The latency gap on oolong real was huge, which made debugging and iteration much easier.

### Context rot stress map

This chart shows what happens as distractor ratio and horizon multiplier increase.

![Context rot stress map](/images/rlm-context-rot-heatmap.png)

How to read it:

1. Rows are distractor ratio.
2. Columns are horizon multiplier.
3. Numbers inside cells are task score.
4. Lighter and darker color shifts show how fragile or robust a method is across conditions.

One important caveat is timing. At the moment of writing, two heavy real dataset text buffer shards are still running, so that corner is still filling in.

### Continual long horizon memory

This chart tracks quality as memory capacity changes.

![Continual long horizon capacity](/images/rlm-continual-capacity.png)

What I learned here:

1. In the tested stream setup, capacity changes gave very small deltas.
2. That is useful, because it suggests the retrieval policy was already stable in this regime.
3. It does not prove capacity never matters. It only says this event budget was not enough to create separation.

## Training setup

Even though this is called a continuous learning campaign, this stage did not change model weights.

1. Base weights stayed frozen.
2. Adaptation happened in memory behavior and retrieval logic.
3. Seed and split were fixed for comparability.

I like this setup for diagnosis. It isolates memory mechanics before introducing optimizer noise.

## Ablations and what they taught me

The cleanest ablation in this run was method replacement.

1. Replace Text Buffer with LPM.
2. Hold dataset and model constant.
3. Observe changes in quality, contradiction, and latency.

On sentinel, this gave better quality and better speed in the same direction, which is exactly the outcome I hoped for.

For context rot, the story was more nuanced. Some corners looked robust, some degraded, and one hard real dataset corner is still in progress.

For continual memory, capacity deltas were small in this run. That points to follow up tests with longer streams and more aggressive memory pressure.

## Hypotheses I started with

I started with three hypotheses.

1. LPM should reduce cost while preserving or improving quality.
2. Context rot should hurt quality as distractors and horizon increase.
3. Continual memory should keep high hit rate in long sessions.

After this run, hypothesis one looks strong.

Hypothesis two is partly confirmed and still incomplete for one hard corner.

Hypothesis three is promising on the tested stream, with more stress needed to find limits.

## Concrete examples from outputs

A few examples made the behavior easy to feel.

1. Some count questions were answered exactly, such as spells cast by Caleb and rolls by Jester.
2. Some hard count queries still returned unknown.
3. In context rot synth, the model could correctly identify least common labels in one item and fail on a neighboring item.

That mix is normal in this kind of stress test, and it is exactly why per example logs matter.

## Timeline

The campaign had a clear rhythm.

1. Sentinel checks finished first.
2. Continual runs finished next.
3. Context rot tail became the bottleneck.
4. Tail sharding across four GPUs recovered throughput.
5. The run moved from one active job to two heavy shard jobs.

The biggest practical lesson is simple. Throughput can collapse at the end of a campaign if scheduling logic does not adapt to heavy tails.

## Next steps

Here is what I plan next.

1. Finish the two remaining context rot shards and rebuild final summaries.
2. Compare hard corner behavior with retrieval gating variants.
3. Add prompt compression variants for text buffer latency control.
4. Increase stream length so capacity effects are easier to separate.

If your own project deals with long conversations, memory heavy agents, or long multi step tasks, this workflow is worth copying. The combination of sentinel checks, context rot stress, and continual streams gives a very practical map of where memory systems hold and where they crack.
