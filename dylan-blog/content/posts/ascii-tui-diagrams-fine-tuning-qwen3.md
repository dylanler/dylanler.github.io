+++
title = 'The Return of ASCII Art: Fine-Tuning a Small LLM to Think in Terminal Diagrams'
date = 2026-02-06T11:00:00-08:00
draft = false
tags = ["AI", "LLM", "fine-tuning", "QLoRA", "ASCII-art", "terminal", "education", "synthetic-data"]
+++

In an era of photorealistic AI-generated images, I trained a language model to draw with box-drawing characters and pipe symbols.

This isn't nostalgia. It's a bet that the most universal visual medium for AI isn't pixels -- it's text.

## Why ASCII Diagrams Still Matter

Every developer, every terminal session, every SSH connection, every log file, every README -- text is the one output format that works everywhere. No rendering engine, no GPU, no browser required. Just characters on a screen.

But there's a deeper reason. When you force a model to explain photosynthesis using nothing but `─`, `│`, `┌`, `└`, `→`, and monospace text, you're forcing it to **think structurally**. It can't hide behind pretty gradients. The explanation has to be clear enough that box-drawing characters carry the meaning.

This experiment fine-tunes Qwen3-0.6B to generate educational ASCII art and TUI (Terminal User Interface) diagrams for science and engineering topics -- and the results reveal something interesting about how small models learn visual reasoning through text.

## The Experiment

### Architecture

The pipeline mirrors my [p5.js physics experiment](/posts/fine-tuning-qwen3-p5js-physics-animations/) but targets a fundamentally different output space:

- **100 parallel agents** generate 1,000 synthetic training examples
- **QLoRA** (4-bit quantized LoRA) fine-tuning on 4x A100 GPUs
- **Qwen3-0.6B** as the base model
- **Three diagram styles**: `ascii_art`, `tui_flow`, `hybrid`

### What the Model Learns to Generate

Given a prompt like "Explain the water cycle," the model produces something like:

```
    ☀ SOLAR ENERGY
        │
        ▼
┌──────────────────────────┐
│   EVAPORATION            │
│   Lakes, oceans, rivers  │──→ Water vapor rises
│   Heat converts liquid   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   CONDENSATION           │
│   Cool air at altitude   │──→ Clouds form
│   Vapor → tiny droplets  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   PRECIPITATION          │
│   Rain, snow, hail       │──→ Falls to surface
│   Gravity pulls water    │
└────────────┬─────────────┘
             │
             ▼
    ≈≈≈ COLLECTION ≈≈≈
    Streams → Rivers → Ocean
             │
             └──→ (cycle repeats)
```

This isn't just decoration. The spatial layout communicates the sequential, cyclical nature of the process in a way that prose alone cannot.

### The Agent Persona System

The most creative engineering decision: each of the 100 generation agents gets assigned one of **8 distinct personas**:

1. **Terminal-native science teacher** -- clean box-drawing layouts
2. **Systems engineer** -- architecture diagram style with flows
3. **Physics explainer** -- intuition-first with force arrows
4. **Biology educator** -- lifecycle timelines with stages
5. **Chemistry visualizer** -- molecular structures with bonds
6. **Network diagram specialist** -- node-and-edge thinking
7. **Retro computing enthusiast** -- DOS/BBS aesthetic
8. **Data visualization minimalist** -- sparklines and compact charts

This creates natural diversity in the training data. The same topic (say, DNA replication) gets visualized as a timeline by one persona, a flowchart by another, and a side-by-side comparison by a third. The model learns that there are multiple valid ways to represent any concept.

## Training Results

| Metric | Value |
|--------|-------|
| Training examples | 1,000 |
| Training steps | 45 |
| Epochs | 3 |
| Initial train loss | 4.13 |
| Final train loss | **0.11** |
| Loss reduction | **97.4%** |
| Initial eval loss | 2.06 |
| Final eval loss | **0.12** |
| Quantization | 4-bit (nf4) |
| LoRA rank | 32 |

The 97.4% loss reduction is dramatic. The model essentially memorizes the ASCII diagram generation pattern within 45 steps. But here's the interesting part: the eval loss drops to 0.12 as well, meaning it's not just memorizing -- it's learning transferable patterns.

### QLoRA vs Full LoRA

This experiment uses QLoRA (4-bit quantization + LoRA) compared to the p5.js experiment's standard LoRA. The tradeoff:

| Approach | Memory | Speed | Quality |
|----------|--------|-------|---------|
| Full LoRA (p5.js experiment) | ~16GB/GPU | 2.9 min | 85.6% token accuracy |
| QLoRA (this experiment) | ~4GB/GPU | Similar | 97.4% loss reduction |

QLoRA's 4x memory reduction means this could run on consumer GPUs. A single RTX 4090 could handle training. The quality tradeoff is minimal for this domain because ASCII art has lower entropy than JavaScript -- there are fewer valid next tokens at any point, so quantization losses matter less.

## Three Insights That Surprised Me

### 1. ASCII Art is a Compression Language

Think about what the model is actually learning. An ASCII diagram of photosynthesis contains:
- **Spatial relationships**: What's above/below/connected to what
- **Process flow**: Arrows showing directionality
- **Grouping**: Boxes that cluster related concepts
- **Labels**: Text anchored to visual elements
- **Hierarchy**: Nesting depth indicates abstraction level

This is essentially a **visual programming language for explanations**. The model learns to compile natural language into this structured visual format. It's lossy compression -- you can't capture everything about photosynthesis in 30 lines of ASCII -- but the compression forces prioritization of the most important relationships.

### 2. Personas Create Better Diversity Than Temperature

In synthetic data generation, the standard approach to diversity is cranking up the temperature. But high temperature produces noise. The persona system produces **structured diversity** -- each persona has a coherent visual philosophy that generates internally consistent but mutually distinct examples.

A systems engineer persona will never generate a timeline. A biology educator won't produce a network diagram. But both create valid, useful representations. This is closer to how real educational content works: different teachers explain the same concept differently, not randomly, but according to their mental models.

### 3. Text-Based Visual Reasoning Transfers

The most speculative insight: training a model to generate ASCII diagrams might improve its general reasoning about spatial and structural relationships. When the model learns that "DNA replication" maps to a fork-shaped diagram with parallel arrows, it's encoding a structural understanding that could transfer to other tasks.

This is analogous to how learning to draw improves observational skills in humans -- the act of representing forces you to understand.

## The Broader Argument: Text-First AI Visualization

We're building increasingly sophisticated multimodal AI systems. But there's a case for text-based visualization as a **first-class output modality**:

**Accessibility**: Works in screen readers, terminal emulators, email, logs, chat
**Reproducibility**: Deterministic rendering -- what you generate is exactly what anyone sees
**Editability**: Humans can manually tweak ASCII diagrams; you can't easily edit a PNG
**Composability**: Embed diagrams in code comments, documentation, commit messages
**Speed**: No rendering pipeline, no image generation model, instant output
**Debuggability**: You can read the "image" as text, character by character

For educational AI, this means:
- Students in low-bandwidth environments get visual explanations
- Terminal-based tutoring systems work over SSH
- Diagrams embed naturally in Jupyter notebooks, README files, and chat interfaces
- Teachers can modify and annotate generated diagrams

## What This Shares with the p5.js Experiment

Both experiments follow the same pattern:

```
Domain Expert Model (Claude/GPT) generates training data
  → Parallel agents create diverse synthetic examples
  → Small model (Qwen3-0.6B) fine-tuned with LoRA/QLoRA
  → Deployed for fast, cheap, domain-specific generation
```

But the output spaces are complementary:

| | p5.js Animations | ASCII Diagrams |
|--|-----------------|----------------|
| **Medium** | Interactive browser canvas | Static terminal text |
| **Requires** | Browser + JavaScript runtime | Any text display |
| **Strength** | Dynamic, visual, engaging | Universal, embeddable, accessible |
| **Best for** | Live demonstrations | Documentation, explanations |
| **Offline?** | Needs browser | Works anywhere |

Together, they suggest a future where small, specialized models handle different "rendering backends" for education -- the same physics concept explained as an interactive simulation OR a terminal diagram, depending on context.

## Running It

```bash
git clone https://github.com/dylanler/ascii-tui-qwen3-lab
cd ascii-tui-qwen3-lab

# Full pipeline: generate data → train → plot → sample
bash scripts/run_full_experiment.sh

# Or step by step:
uv run python scripts/generate_synthetic_dataset.py  # 100 agents, 1000 examples
torchrun --nproc_per_node=4 scripts/train_qwen3_ascii_tui.py  # QLoRA fine-tune
uv run python scripts/generate_samples.py  # Generate test outputs

# Serve via vLLM
bash scripts/start_vllm_endpoint.sh
```

The trained LoRA adapter is published on Hugging Face at `mr-dee/qwen3-ascii-tui-lora`.

## Future Directions

1. **Mermaid.js bridge**: Generate ASCII diagrams, then auto-convert to Mermaid for richer rendering when available
2. **Interactive TUI**: Build a curses-based application where the model generates and animates ASCII diagrams in real-time
3. **Multi-model pipeline**: Small model generates ASCII diagram → vision model evaluates visual quality → feedback loop
4. **Accessibility testing**: Partner with screen reader users to evaluate whether ASCII diagrams actually improve understanding
5. **Diagram-to-code**: Reverse direction -- given an ASCII diagram, generate the code that implements the depicted system

The most ambitious direction: **a universal text-based visualization model** that can render any concept as a terminal-friendly diagram. Not as a gimmick, but as the most portable, accessible, and debuggable way for AI to show its reasoning.

---

*Source code: [github.com/dylanler/ascii-tui-qwen3-lab](https://github.com/dylanler/ascii-tui-qwen3-lab)*

*The future of AI visualization might not be more pixels. It might be better characters.*
