+++
title = 'Does the harness evolve first, or the model?'
date = 2026-10-02T20:55:00-07:00
draft = false
math = true
tags = ["AI", "agents", "evolution", "tools"]
+++

In this toy, the harness evolves first while the brain is primitive. Past a capacity threshold the brain takes over. A greedy brain that edits the tool it thinks is worst can lock in a worse limb than blind selection keeps.

This is a synthetic evolutionary simulation, not a user study and not a test of a real language model. The controller is a small lookup table. Code, protocol, and the raw series: [dylanler/harness-vs-brain](https://github.com/dylanler/harness-vs-brain). Recorded run: 8 seeds, 48 generations, population 32.

## The question

Primitive animals had sensors, contractile cells, and limbs under selection before they had a brain that could redesign those parts. Sponges coordinate feeding and whole-body contractions with no neurons (Musser and colleagues, *Science*, 2021). Brooks argued that evolution spent most of its time on mobility and sensing, and that central problem-solving looks easy only after that base exists ([Intelligence without representation](https://people.csail.mit.edu/brooks/papers/representation.pdf), 1991). Sims coevolved bodies and the circuits that drive them ([Evolving virtual creatures](https://dl.acm.org/doi/10.1145/192161.192167), SIGGRAPH 1994). Cheney, Bongard, SunSpiral, and Lipson later called co-optimizing morphology and control unusually hard.

The agent version: a capacity-limited controller (the brain) sits inside a harness of tools and sensors (the limbs). Should the brain decide which tools to keep, or should the harness change under blind variation and selection?

Toolformer, Gorilla, and HuggingGPT are model-first: the tool menu is given. Voyager keeps the model frozen and grows a skill library under environment feedback. The Darwin Gödel Machine edits agent code and keeps a change only if the benchmark improves, and notes that more tools are not automatically helpful. None of those is the comparison below. The comparison is a \(K\)-slot table, not a transformer.

## Setup

Six resource niches. Three are common, three are rare. A catalog of 14 tools (bare hands, a cheap generalist, six specialists, a bait tool that looks good on common niches, a key tool that pays on rare niches, and junk) plus one sensor per feature. A tool that is off cannot be used. A sensor that is off returns no information.

The controller has \(K\) prototype slots. Each slot stores a feature prototype and one preferred tool. The agent picks the nearest prototype on the sensed coordinates and uses that slot's tool if it is installed. Otherwise it falls back to bare hands. \(K = 1\) is one default action. \(K = 8\) can in principle store one mapping per niche.

Three tasks, same for every regime: foraging with partial sensing, a contextual tool-choice bandit, and a delayed-reward landscape. On the delayed task, 10 common steps are followed by 2 rare jackpot steps at reward scale 5. A myopic score sees only the common slice.

Per step, if the patch has a niche, success probability is capability times reliability:

$$
p = \mathrm{cap}(\mathrm{tool}, \mathrm{niche}) \cdot (1 - \nu_{\mathrm{tool}})
$$

On an empty patch, bare hands score \(0.42\) and any other tool scores \(0.10\). The step score is \(p\) times a reward scale, minus a small use fee.

Harness cost charges installed tools, a tax on tools that almost never fail, and each sensor:

$$
C = \sum_j c_j \mathbf{1}_{j \in H} + 0.35 \sum_j \mathrm{clip}(0.22 - \nu_j,\, 0,\, 0.20) + c_s \, n_{\mathrm{sensors}}
$$

Spam is junk past a full kit:

$$
P = 0.10 \cdot \max(0,\, n_{\mathrm{tools}} - 8) + 0.04 \cdot \max(0,\, n_{\mathrm{sensors}} - n_{\mathrm{features}})
$$

Fitness is mean task score minus those penalties. The weights in the recorded run are \(0.32\) on cost and \(0.10\) on spam:

$$
F = S - 0.32\, C - 0.10\, P
$$

\(S\) is the mean of the three task scores. The brain-directed arm is selected on a myopic twin of \(F\), computed from common steps only. True \(F\) is still recorded.

Five regimes, same evaluation budget:

1. **Model-first.** Freeze a complete field kit. Mutate the controller. Select on true \(F\).
2. **Harness-first.** Freeze one random controller of capacity \(K\). Mutate tools and sensors. Select on true \(F\).
3. **Coevolution.** Mutate both. Select on true \(F\).
4. **Brain-directed.** The myopic value of each installed tool chooses which limb to edit, and selection uses that same short-horizon score. Unused tools on the common slice are treated as dead weight.
5. **Conservative coevolution.** Mutate both, with a lower harness mutation rate and more elites. Select on true \(F\).

## Results

Elite fitness is the mean true fitness of the top 8 individuals, then the mean across 8 seeds, \(\pm 1\) standard deviation.

![Elite fitness versus generation at K=1 and K=8, one line per regime.](/images/hvb_fitness_vs_generation.png)

**Tight brain.** At \(K = 1\), harness-first elite fitness is \(0.616 \pm 0.051\). Model-first is \(0.148 \pm 0.025\). A one-slot brain cannot represent a context-to-tool map, so evolving the controller on a frozen complete kit goes nowhere. Evolving the kit around that dumb default finds a cheap one-tool body. The limb does the work the brain cannot.

**Past the threshold.** At \(K = 8\), model-first is \(0.697 \pm 0.017\) and harness-first is \(0.591 \pm 0.105\).

![Final elite fitness against controller capacity. Model-first crosses harness-first between K=2 and K=4.](/images/hvb_capacity_sweep.png)

| \(K\) | Model-first | Harness-first |
| --- | --- | --- |
| 1 | \(0.148 \pm 0.025\) | \(0.616 \pm 0.051\) |
| 2 | \(0.320 \pm 0.025\) | \(0.522 \pm 0.140\) |
| 4 | \(0.596 \pm 0.083\) | \(0.573 \pm 0.104\) |
| 8 | \(0.697 \pm 0.017\) | \(0.591 \pm 0.105\) |

The curves cross between \(K = 2\) and \(K = 4\).

**Blind selection versus a myopic editor.** At \(K = 8\), where a controller can reserve a tool for rare contexts, blind harness-first deceptive score is \(0.981 \pm 0.182\). Brain-directed is \(0.677 \pm 0.189\). Elite key-tool retention is \(0.375\) versus \(0.000\). The editor treats the key as dead weight on common steps and drops it. Blind mutation plus true fitness does not systematically hunt it.

At \(K = 1\) the deceptive gap also favored blind selection (\(0.941\) versus \(0.581\)), but both arms dropped the key. A one-slot brain cannot save it for the jackpot, so that cell is a weaker test.

![Two ways to change a limb, plus the H1, H2, and H3 bars.](/images/hvb_harness_vs_brain.png)

**Coevolution churns.** At \(K = 1\), coevolution (\(0.664\)) beat both single-sided arms. At \(K = 8\) it beat harness-first (\(0.685\) versus \(0.591\)) and did not beat model-first (\(0.697\)). Those two high-\(K\) means overlap within a standard deviation. Do not read this as "coevolution always wins."

Open coevolution turned tools over faster than conservative selection: Jaccard distance \(0.122\) versus \(0.079\) at \(K = 8\), and \(0.099\) versus \(0.067\) at \(K = 1\). Conservative selection paid some fitness at \(K = 1\) (\(0.556\) versus \(0.664\)) and was roughly tied at \(K = 8\) (\(0.660\) versus \(0.685\)).

![Elite tool-set turnover versus generation.](/images/hvb_tool_turnover.png)

![Fraction of elites still carrying the delayed-reward key.](/images/hvb_key_retention.png)

Open coevolution often collapsed to a one- or two-tool body and dropped the key. A costly generalist limb can collect the delayed jackpot without a specialized key. That is why key retention is not the only story in the last figure.

## What this does not show

It does not show that a real language-model agent should freeze the weights and evolve tools, or the reverse. The tasks are three synthetic landscapes. The result is narrower: in this toy, the harness is the thing to evolve while the brain is primitive; past a capacity threshold the brain takes over; and a greedy brain that edits the limb it dislikes can lock out a delayed-reward tool that blind selection would sometimes keep.

Reproduce with `python -m sim.run` from the repo. A `--smoke` flag is a tiny budget and is not this result.
