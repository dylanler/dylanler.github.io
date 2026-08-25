+++
title = 'Worlds That Teach Back'
date = 2026-04-08T19:16:00-07:00
draft = false
tags = ["AI", "physics simulation", "embodied learning", "creativity", "discovery"]
+++

A child drops a spoon from a table. Gravity answers immediately.

The answer is not a paragraph. It is a collision, a sound, and a new expectation about what happens next. The world teaches through resistance.

That simple fact sits behind several experiments in this repository. One proposed a developmental curriculum inside MuJoCo. Another trained Qwen3 0.6B to generate interactive p5.js physics animations. Together they point toward a strange and fertile idea: perhaps a model should not only read descriptions of reality. Perhaps it should build small realities, touch them, and be corrected by what they do.

## From corpus to playground

Text is generous. It lets an incorrect claim remain elegant. Physics is less polite. A bridge falls. A pendulum gains energy when it should lose it. A ball passes through the floor. The simulation turns a conceptual mistake into an observable event.

The developmental learning proposal organizes this feedback into stages. First come simple actions and object permanence. Then tools, obstacles, transfer, and social behavior. This is not merely a longer prompt. It is a curriculum where later discoveries depend on earlier physical intuitions.

The p5.js experiment approaches the same frontier from the opposite direction. Instead of placing a model inside a simulator, it asks the model to write one. One hundred parallel agents generated one thousand training examples. QLoRA adapted a small model on four A100 GPUs. The target was code, but the real subject was causality.

{{< frontier mode="world" id="april-world" >}}

Change gravity and surface drag, then release another learner. Nothing in the scene is intelligent. That is the point. The world supplies a stable consequence, which is exactly what a learning system can push against.

## Code as compressed imagination

An animation program is a peculiar kind of sentence. It describes not just what a scene looks like, but how the scene will continue to change.

That makes code generation an unusually rich creative test. A model must coordinate geometry, time, state, and visual explanation. If it writes a pendulum, the length must constrain the path. If it writes an orbit, velocity and attraction must remain in conversation. The program becomes a hypothesis about a possible world.

The repository also explores terminal diagrams made from ASCII characters. At first this seems like a separate nostalgic experiment. It is actually another version of the same discipline. Severe constraints reveal structure. When pixels are unavailable, the model must express relationships with lines, boxes, labels, and space. When prose is insufficient, executable code must carry the explanation through time.

## A curriculum of wonder

The obvious use for generated simulations is education. Ask for buoyancy, orbital mechanics, or wave interference and receive a manipulable scene. The more radical use is education for the model itself.

Imagine a loop with four steps.

1. The model predicts what a world will do.

2. It writes or chooses an action.

3. The simulator returns the consequence.

4. The model updates its explanation.

This is discovery in miniature. The system is no longer rewarded only for sounding like the archive. It is rewarded for surviving contact with a world.

## The open question

Simulation can also become a comfortable illusion. A learner may master the quirks of one engine without acquiring concepts that transfer. Beautiful generated animations may conceal incorrect equations. A curriculum may reward shortcuts invisible to its designer.

So the next experiments need crossings. Change the renderer while preserving the law. Change the object while preserving the relation. Move from p5.js to MuJoCo, from text diagrams to motion, and ask which concepts survive.

The most exciting result would not be a perfect simulation. It would be a model that notices when its world is wrong.

We have spent decades building machines that answer questions about maps. The repository’s physics work suggests another path. Give the machine a territory small enough to build, strange enough to explore, and honest enough to push back.
