+++
title = 'What If LLMs Remembered in Vectors Instead of Words?'
date = 2026-02-25T14:00:00-08:00
draft = false
tags = ["AI", "LLM", "memory", "latent-space", "long-context", "experiment", "transformers", "RLM", "recursive-language-models"]
+++

What happens when you give a language model a 60,000 token document and ask it a question about paragraph 47?

It forgets. Or worse, it makes something up.

This is the long context problem and it is one of the most important open challenges in language modeling today. Context windows keep growing (Gemini has 1M tokens, Claude has 200K) but models still struggle with information buried deep in the middle of long inputs. Retrieval augmented generation helps for lookup style queries but falls apart when the answer requires synthesizing information across multiple sections.

The Recursive Language Model (RLM) approach offers one fix: chop the document into chunks, summarize each chunk into text, glue the summaries together, and feed that back to the model. But every time you squeeze information through the vocabulary bottleneck (turning hidden states into words), you lose something. Nuance. Uncertainty. The subtle distributional signals that a transformer builds up internally but can never fully express as tokens.

So I ran an experiment: what if instead of summarizing each chunk into text, we saved the model's raw hidden states as compressed vectors and let a neural network figure out how to use them later?

I called it **Latent Pager Memory**. The name comes from virtual memory paging in operating systems. Instead of paging text to disk, you page latent states. The full code is at [github.com/dylanler/rlm-experiment-claude](https://github.com/dylanler/rlm-experiment-claude).

## Why This Matters: The Information Bottleneck Problem

To understand why latent paging might help, you need to understand what happens when a transformer processes text. At each layer, the model builds up a rich internal representation. By the time you reach the final layers, each token position contains a 2048 dimensional vector (in our case with Qwen3-1.7B) that encodes not just the token itself but its relationships to everything else in the context.

When the standard RLM approach asks the model to "summarize this chunk," it forces all of that rich 2048 dimensional information through a vocabulary bottleneck of discrete tokens. Information gets destroyed. The model has to decide what to keep and what to throw away, and it has to express everything in words.

Here is the key insight: **some information that transformers encode internally cannot be faithfully expressed in text.** Uncertainty signals, implicit relationships between entities, the degree of confidence in different facts. These live in continuous vector space and get lost when you force them into discrete tokens.

The latent pager skips this bottleneck entirely.

## How It Works

The system has four trainable components sitting on top of a frozen Qwen3-1.7B:

```
Document → Chunk (1024 tokens, 128 overlap)
    → Frozen Qwen3-1.7B forward pass
    → Extract hidden states from layers [7, 14, 21, 27]
    → Last-token pooling → [4 × 2048] = 8192 dim per chunk

8192-dim snapshot → PageCompressor → 512-dim page vector (16× compression)
    → Store in PageStore

All page vectors → PageAggregator (Perceiver cross-attention, 16 queries)
    → 16 soft prompt tokens [16 × 2048]
    → Prepend to question embeddings
    → Frozen LM generates answer
```

| Component | Parameters | What It Does |
|---|---|---|
| **PageCompressor** | 9.4M | Linear(8192→512) + SiLU + LayerNorm. Compresses 16x. |
| **PageAggregator** | 82.2M | Perceiver style cross attention. 16 learnable query tokens attend over variable length page sequences. |
| **Base Model (Frozen)** | 1.7B | Qwen3-1.7B. Never updated during training. |
| **Total Trainable** | 91.6M | Only 5.4% of the total parameter count. |

The baseline (Text Buffer / RLM) does something different: it runs the LM to generate a text summary for each chunk, concatenates all the summaries, and runs the LM again to generate a final answer. More LM generation calls, more places where information gets destroyed through the vocabulary bottleneck.

## The Setup

| Detail | Value |
|---|---|
| GPU | 4× NVIDIA A100-SXM4-80GB |
| Base Model | Qwen/Qwen3-1.7B (frozen, bfloat16) |
| Hidden Size | 2048 |
| Layers | 28 |
| Training Samples | 2,000 |
| Validation Samples | 300 |
| Test Samples | 500 |
| Document Length | 8K to 65K tokens |
| Task Types | Single fact extraction (52%), Multi-hop reasoning (48%) |
| Source | Mixed (Wikipedia, arXiv, news articles) |

## The Results

Here is the headline after three iterations of trying:

![Metrics Comparison](/images/lpm-metrics-comparison.png)

| Metric | Text Buffer (Baseline) | Latent Pager | Change | p-value | 95% CI |
|---|---|---|---|---|---|
| **F1** | 0.0182 | **0.0257** | +41.5% | < 0.001 | [0.0048, 0.0103] |
| **ROUGE-L** | 0.0177 | **0.0260** | +47.0% | < 0.001 | [0.0057, 0.0109] |
| **Hallucination Rate** | **0.292** | 0.580 | +98.4% | < 0.001 | [0.253, 0.321] |
| **Avg Latency** | 19.55s | **7.65s** | 2.55× faster | | |
| **Peak Memory** | **1.02 GB** | 1.82 GB | +77% | | |
| **Exact Match** | 0.000 | 0.000 | — | | |

All differences were statistically significant at p < 0.001 using 10,000 paired bootstrap iterations.

The latent pager is genuinely better at answering questions (higher F1 and ROUGE-L) and genuinely faster (because it doesn't need to generate text summaries for each chunk). But it hallucinates way more. Almost double the hallucination rate.

This is the central tension of the experiment: the model gets closer to the right answer more often, but when it is wrong, it is wrong with high confidence and fabricated details.

### Speed Advantage

![Latency Comparison](/images/lpm-latency.png)

The speed difference is dramatic and easy to explain. The text buffer baseline needs to run `model.generate()` for every chunk (to produce summaries) and then again for the final answer. That is multiple expensive autoregressive generation passes. The latent pager only does forward passes through the frozen model (no generation, just hidden state extraction) and one final generation. Forward passes are much cheaper than autoregressive decoding.

### Performance by Task Type

![Per-Task Breakdown](/images/lpm-per-task.png)

| Task | Metric | Baseline | Latent Pager | Improvement |
|---|---|---|---|---|
| Single Fact Extraction (260) | F1 | 0.0206 | 0.0314 | +52% |
| Single Fact Extraction (260) | ROUGE-L | 0.0210 | 0.0323 | +54% |
| Single Fact Extraction (260) | Hallucination | 0.317 | 0.662 | +109% (bad) |
| Multi-Hop Reasoning (240) | F1 | 0.0155 | 0.0195 | +26% |
| Multi-Hop Reasoning (240) | ROUGE-L | 0.0142 | 0.0192 | +35% |
| Multi-Hop Reasoning (240) | Hallucination | 0.265 | 0.491 | +85% (bad) |

Two patterns stand out. First, the latent pager helps more on single fact extraction (+52%) than multi-hop reasoning (+26%). This makes sense because single fact lookup is closer to information retrieval from compressed states, while multi-hop requires combining facts across chunks, which is harder to do through soft prompts.

Second, hallucination is worse across both task types but especially bad for single fact extraction (0.662). When the model has a compressed page vector that vaguely relates to the question, it often generates a confident but fabricated answer rather than saying "I don't know."

## The Three Iterations (And Why Simpler Won)

I did not get here on the first try. In fact the first two attempts failed outright.

![Three Iterations](/images/lpm-three-iterations.png)

**Version 1** used the initial hyperparameters I picked: mean pooling, 32 soft tokens, 2 aggregator layers, learning rate 1e-4. The result was F1 of 0.0136, which is *worse* than the baseline of 0.0182. The model had too many parameters (120M trainable) for the amount of training data (2,000 samples) and the wrong pooling strategy was destroying information.

Then I ran ablation studies. I swept across pooling strategies, number of soft tokens, aggregator depth, compression dimension, and extraction layers. The ablations revealed something that changed everything: three individual settings each independently beat the baseline on their own.

| Setting | F1 | vs Baseline | What Changed |
|---|---|---|---|
| last_token pooling (vs mean) | 0.0231 | +27% | Pooling strategy |
| 16 soft tokens (vs 32) | 0.0240 | +32% | Query token count |
| 1 aggregator layer (vs 2) | 0.0232 | +27% | Model depth |

The original model was being held back by bad hyperparameters. Not a bad architecture, bad settings.

**Version 2** went too far in the other direction. I added question conditioned aggregation (a bottleneck projection that biases the aggregator based on the question) and a reconstruction auxiliary loss (forcing page vectors to be able to reconstruct original hidden states). Both sounded smart on paper. Both made things worse. Test F1 dropped to 0.0143.

Why? The question conditioning added 4.5M extra parameters that overfitted on the small training set. The model learned to associate specific question patterns with specific page configurations, but this did not generalize. The reconstruction loss pulled the training gradient away from the actual objective (answering questions correctly) and toward a proxy objective (reconstructing hidden states) that turned out to be only loosely correlated.

**Version 3** was the simplest: just apply the ablation optimal settings (last_token pooling, 16 soft tokens, 1 aggregator layer), use the pretrained compressor, and keep everything else minimal. No question conditioning. No reconstruction loss. This version reached test F1 of 0.0257.

The lesson was clear. On a small dataset (2,000 training samples) with a small model (1.7B parameters), every extra parameter you add is a parameter that overfits. Simpler wins.

## The Ablation Findings

These are the results that actually guided the final design. Each ablation trained for 5 epochs and was evaluated on 50 validation samples.

### Pooling Strategy: The Single Biggest Lever

![Pooling Comparison](/images/lpm-ablation-pooling.png)

| Strategy | F1 | Hallucination | Train Loss |
|---|---|---|---|
| Mean pooling | 0.0191 | 0.273 | 3.989 |
| **Last token** | **0.0231** | **0.073** | **3.505** |

This was the single most important design decision. Last token pooling gave a 21% F1 boost and reduced hallucination by 73%.

Why is last token so much better? Think about how attention works in a transformer. By the final layer, the last token position has attended over the entire sequence. Its hidden state is essentially the model's own internal summary of everything it just read. When you take the mean across all positions, you dilute this concentrated signal with positions that only contain local information.

This has broader implications for anyone doing feature extraction from transformers. If you are pulling representations for downstream tasks, try last token pooling before mean pooling.

### Number of Soft Tokens

![Soft Token Ablation](/images/lpm-ablation-soft-tokens.png)

| Tokens | F1 | Hallucination | Aggregator Params |
|---|---|---|---|
| 8 | 0.0186 | **0.211** | ~41M |
| **16** | **0.0240** | 0.271 | ~82M |
| 32 | 0.0191 | 0.273 | ~164M |
| 64 | 0.0171 | 0.316 | ~328M |
| 128 | 0.0163 | 0.261 | ~656M |

16 tokens is the sweet spot. Below 16, there is not enough bandwidth to carry the compressed document information. Above 16, the aggregator's parameter count grows linearly with the number of query tokens and overfitting kicks in.

Notice the U-shaped hallucination curve: 8 tokens has the lowest hallucination (0.211) because it carries so little information that the model stays cautious. 64 tokens has the highest (0.316) because the model has enough bandwidth to generate confident but unfaithful outputs.

### Compression Dimension (d_page)

![d_page Ablation](/images/lpm-ablation-dpage.png)

| d_page | F1 | Hallucination | Compression Factor |
|---|---|---|---|
| 128 | 0.0185 | 0.361 | 64× |
| 256 | 0.0153 | **0.240** | 32× |
| **512** | **0.0191** | 0.273 | **16×** |
| 1024 | 0.0161 | **0.232** | 8× |
| 2048 | 0.0179 | 0.356 | 4× |

There is no clean monotonic relationship here. 512 gives the best F1 at 16× compression. Interestingly, the lowest hallucination rates come from the middle dimensions (256 and 1024), not the extremes. My interpretation: heavy compression (128, 64×) loses too much information and the model hallucinates to fill gaps. Light compression (2048, 4×) preserves noise and irrelevant details that confuse the aggregator.

### Aggregator Depth

![Aggregator Depth](/images/lpm-ablation-depth.png)

| Depth | F1 | Hallucination | Train Loss |
|---|---|---|---|
| **1 layer** | **0.0232** | 0.330 | 3.865 |
| 2 layers | 0.0191 | 0.273 | 3.989 |
| 4 layers | 0.0181 | **0.194** | 3.827 |

One layer gives the best F1. But look at the tradeoff: 4 layers has the lowest hallucination (0.194) even though its F1 is worst. Deeper aggregators learn to be more cautious. They produce less confident outputs, which means fewer hallucinations but also fewer correct answers.

This is an interesting design axis for future work. In applications where faithfulness matters more than accuracy (medical, legal), deeper aggregators might be preferred despite lower F1.

## Training Dynamics

![Training Curves](/images/lpm-training-curves.png)

| Epoch | Train Loss | Val Loss | Val F1 | LR | Note |
|---|---|---|---|---|---|
| 1 | 3.581 | 3.102 | 0.0238 | 2.94e-4 | |
| **2** | **3.321** | **3.039** | **0.0294** | **2.74e-4** | **Best checkpoint** |
| 3 | 3.332 | 3.020 | 0.0266 | 2.41e-4 | |
| 4 | 3.208 | 3.096 | 0.0233 | 1.99e-4 | |
| 5 | 3.166 | 3.028 | 0.0217 | 1.52e-4 | |
| 6 | 3.132 | 3.034 | 0.0183 | 1.05e-4 | F1 drops to baseline |
| 7 | 3.106 | 3.029 | 0.0189 | 6.3e-5 | |
| 8 | 3.084 | 3.022 | 0.0200 | 3.0e-5 | |
| 9 | 3.072 | 3.023 | 0.0167 | 3.0e-5 | Lowest F1 |
| 10 | 3.067 | 3.025 | 0.0191 | 3.0e-5 | |

The best model comes from epoch 2. After that, F1 drops continuously even as train loss keeps decreasing. This is classic overfitting on a small dataset.

A critical lesson here: **do not select your checkpoint by validation loss.** Val loss keeps decreasing through epoch 8 while val F1 peaks at epoch 2 and falls. If I had used val loss for checkpoint selection (which is the default in many training frameworks), I would have picked a much worse model. Always select by the metric you actually care about.

## The Repetition Penalty Breakthrough

There is a detail that is not glamorous but made a huge difference. After training version 3 and evaluating it, the test F1 was 0.013. Way worse than the 0.029 validation F1. I looked at the predictions and found the model was generating repetitive loops:

```
The answer is \boxed{Napoleon}. The answer is \boxed{Napoleon}. The answer is
\boxed{Napoleon}. The answer is \boxed{Napoleon}. The answer is \boxed{Napoleon}...
```

The model had learned to output the right content but couldn't stop. The soft prompt tokens put the model into a "mode" where it wanted to repeat, and without any penalty for repetition it just kept going until max tokens.

Two fixes solved this:

1. Add `repetition_penalty=1.3` to the generate call
2. Post process outputs to truncate at the first repeated sentence

After these changes, test F1 jumped from 0.013 to 0.0257. **More than half the final performance came from fixing generation quality, not architecture changes.** This is a lesson I keep relearning: boring engineering decisions often matter more than clever architecture.

## What the Predictions Actually Look Like

Both systems struggle with this task. Qwen3-1.7B is simply too small to reliably answer complex questions about long documents. But the failure modes are different and revealing.

### Example 1: Baseline Gets It Right, Latent Pager Hallucinates

**Question:** "Where did the hijacked plane rammed by Heather Penney crash?"
**Gold answer:** Pennsylvania

| System | Prediction | F1 | Halluc. |
|---|---|---|---|
| **Latent Pager** | "The 1980s. The plane crashed in the ocean off of a small island near where Heather Penney was born... Answer: D: Texas" | 0.0 | 0.60 |
| **Baseline** | "The hijacked plane rammed by Heather Penney crashed into a field in Pennsylvania." (repeated 5x) | 0.011 | 0.46 |

The baseline got the right answer but buried it in repetition. The latent pager fabricated a confident wrong answer with specific details that do not exist.

### Example 2: Latent Pager Gets It Right, Baseline Hallucinates

**Question:** "Which has more species, Festuca or Dactylis?"
**Gold answer:** Festuca

| System | Prediction | F1 | Halluc. |
|---|---|---|---|
| **Latent Pager** | "Festuca has more species. 1057 different types of Festuca..." | 0.028 | 0.67 |
| **Baseline** | "Based on the information provided, the answer is Dactylis... [long wrong reasoning]" | 0.015 | 0.11 |

The latent pager got the right direction but hallucinated a specific number. The baseline got it wrong but with lower confidence.

### Example 3: Both Wrong, Different Ways

**Question:** "What type of tours does the host of Tosh.0 conduct?"
**Gold answer:** stand-up comedy

| System | Prediction | F1 | Halluc. |
|---|---|---|---|
| **Latent Pager** | "Tours of the world's most famous places. 1980s, Tosh and his wife took a tour to Europe..." | 0.0 | 1.0 |
| **Baseline** | "The host of Tosh.0 conducts a (blank) tour." (repeated 12x) | 0.0 | 0.43 |

The latent pager confidently made up an entire narrative. The baseline got stuck in a loop but at least didn't fabricate details.

### Failure Mode Summary

| Failure Mode | Latent Pager | Baseline |
|---|---|---|
| Confabulation (making up facts) | Very common | Rare |
| Repetition loops | Rare (with rep. penalty) | Very common |
| Quiz format hallucination | Common (generates A/B/C/D unprompted) | Rare |
| Self referential meta-commentary | Rare | Common |
| Correct answer buried in noise | Sometimes | Often |

## Why Hallucination Got Worse

This is the most important question. The whole motivation was that latent states should be more faithful than text summaries, so why does the latent pager hallucinate more?

My best theory: the soft prompt injection creates a **modality gap**. The frozen LM was trained on text token embeddings. Every embedding it has ever seen came from its own vocabulary. The soft prompt tokens come from a completely different distribution: the output of a cross attention module over compressed page vectors. The LM does not know what to do with these unusual embeddings, so it falls back on its priors, which means generating plausible sounding text that is not grounded in the actual input.

The text baseline, by contrast, gives the LM text it can actually read and ground its answers in. The summaries are lossy, but at least they are in a format the model understands natively.

This modality gap theory is supported by the ablation data. Last token pooling dramatically reduces hallucination (from 0.273 to 0.073 in ablations) because last token hidden states are closer to the distribution the LM naturally works with. They are more "token-like" than mean pooled representations.

## Hypothesis Scorecard

![Hypothesis Scorecard](/images/lpm-hypothesis-scorecard.png)

Before running the experiment, I registered five hypotheses. Here is how they turned out:

| Hypothesis | Prediction | Actual Result | Verdict |
|---|---|---|---|
| **H1:** Hallucination ≥ 10% reduction | Latent states preserve faithful information | Hallucination went UP 98% | **NOT SUPPORTED** |
| **H2:** Multi-hop F1 ≥ 5 point gain | Cross-chunk aggregation helps reasoning | +26% relative, +0.4 absolute | **SUPPORTED** (weakly) |
| **H3:** Global consistency improves | Latent aggregation enforces coherence | No consistency data collected | **INCONCLUSIVE** |
| **H4:** Retention scales with d_page | More dimensions = more information | Clear capacity/quality tradeoff | **SUPPORTED** |
| **H5:** Compute ≤ 1.5x baseline | Forward passes cheaper than generation | Actually 0.39x (2.55x faster!) | **SUPPORTED** |

3 out of 5 hypotheses supported. But H1 (the central claim) was dead wrong. That is the honest result.

## What I Learned About Building This

Beyond the specific results, this experiment taught me several things that generalize to other ML projects.

**Ablations before complexity.** I wasted time on v2 (adding question conditioning and reconstruction loss) when ablations would have told me the real problem was hyperparameters. Always run ablations on your simplest model before adding complexity.

**Checkpoint selection metrics matter more than you think.** Selecting by val_loss instead of val_f1 would have given me a model from epoch 8 instead of epoch 2. That is a 35% F1 difference from one line of code.

**Generation settings are not an afterthought.** The repetition penalty fix was worth more than any architecture change. If your model generates text, tune your generation parameters as carefully as your model architecture.

**Small data amplifies everything.** With 2,000 training samples, every extra million parameters is an overfitting risk. The jump from v1 (120M params) to v3 (91.6M params) was almost entirely about reducing parameters, not improving the architecture.

**The boring fixes are often the biggest wins.** Pooling strategy, repetition penalty, checkpoint metric. None of these are publishable insights. All of them mattered more than the "interesting" architectural choices.

## The Future of RLM Models: Where This Is All Going

This experiment sits at the intersection of two major trends in LLM research: external memory systems and latent space reasoning. Based on what I learned, here is where I think Recursive Language Models and their descendants are heading over the next few years.

### Prediction 1: Hybrid Text-Latent Memory Will Become Standard

Pure text buffers lose information. Pure latent buffers hallucinate. The obvious next step is a hybrid system that stores both: text summaries for grounding and interpretability, latent page vectors for preserving nuance and uncertainty signals.

The text component provides a "safety net" that keeps the model grounded, while the latent component provides the rich distributional information that text cannot capture. You could imagine an architecture where the model first reads the text summary to establish a factual foundation, then uses the latent page to refine its understanding with the subtle signals that were lost in summarization.

This is already foreshadowed in the ablation results. The 4 layer aggregator had the lowest hallucination rate (0.194) because deeper processing of the latent pages acted as implicit regularization. A hybrid system would make this explicit.

### Prediction 2: Latent Memory Will Shine at 7B+ Scale

Both systems in this experiment got F1 under 0.03. Qwen3-1.7B simply cannot answer most of these questions regardless of how you present the information. The model is too small to have learned enough world knowledge and reasoning ability for complex QA.

At 7B+ scale, models can actually answer questions when given the right context. This changes the dynamics entirely. The text buffer baseline will hit a ceiling because text summaries are inherently lossy regardless of model scale. The latent pager should keep improving because larger models produce richer, more informative hidden states.

I predict that somewhere between 7B and 13B, latent memory systems will achieve both higher accuracy AND lower hallucination than text buffers. The modality gap that caused hallucination in our experiment is partly a small-model problem: larger models have more capacity to interpret novel embedding distributions.

### Prediction 3: LoRA Bridging Will Solve the Modality Gap

The hallucination problem in this experiment comes from injecting foreign embeddings into a frozen model. The model has never seen anything like these soft prompt tokens during training.

LoRA (Low Rank Adaptation) applied specifically to the attention layers that process the soft prompt positions could teach the model to interpret these new embeddings. This is similar to how multimodal models like LLaVA use a projection layer to bridge vision encoders with language models. The same principle applies here: you need a lightweight adapter that translates between the latent page distribution and the text embedding distribution.

This approach keeps the base model's knowledge intact while adding just enough flexibility to process the new input modality. I expect this will reduce hallucination by at least 50% based on analogous results in vision-language alignment.

### Prediction 4: Hierarchical Paging for Ultra-Long Documents

The current system uses flat aggregation: all page vectors are fed into a single cross attention layer. This works for documents with 2-5 chunks but will not scale to documents with 100+ chunks (100K+ tokens).

Future systems will use hierarchical paging: nearby pages get locally aggregated first, then those local summaries get globally aggregated. Think of it like a B-tree for latent states. This preserves local coherence (nearby paragraphs are related) while still allowing global information flow.

The OS paging analogy extends naturally here. Real operating systems use multi-level page tables for efficiency. Latent paging should do the same.

### Prediction 5: Latent Pages As a Universal Memory Format

Right now, every RLM system builds its own memory representation. Text buffers, RAG embeddings, KV-cache compression, latent pages. These are all solving the same problem: how to store what a model learned from text in a way that can be efficiently retrieved later.

I think the field will converge on a standard latent memory format that can be shared across models and tasks. Just like embeddings became a standard interface for retrieval, latent pages (or something like them) will become a standard interface for model memory. You would compute pages once and reuse them across many queries, many models, even many modalities.

The speed advantage we observed (2.55x faster inference) makes this economically compelling. Pre-compute latent pages for your document corpus once, then answer unlimited questions against them at 60% lower cost.

### Prediction 6: The Training Signal Problem Will Drive Innovation

The hardest challenge we faced was training the compressor and aggregator end-to-end. The QA loss provides only ~20 gradient bearing tokens per sample (the short answer). That is an extremely sparse training signal for learning to compress 8192 dimensional hidden states.

Future work will likely develop better training objectives. Self-supervised pretraining on reconstruction (which we tried with mixed results) is one direction. Contrastive learning between page vectors and their source chunks is another. Knowledge distillation from a teacher model that can see the full document is a third.

The reconstruction objective did not work for us because it conflicted with the QA objective. But a staged approach (pretrain for reconstruction, then fine-tune for QA with the reconstruction head frozen) might work better. Our compressor pretraining did help: reconstruction MSE dropped from 375 to 102 over 50 epochs, and the pretrained compressor contributed to v3's success.

## Running the Experiment

The full codebase is available at [github.com/dylanler/rlm-experiment-claude](https://github.com/dylanler/rlm-experiment-claude).

```bash
# Phase 1: Setup and verify environment
python scripts/01_setup_and_verify.py

# Phase 2: Run text buffer baseline
python scripts/02_run_baseline.py

# Phase 3a: Pretrain compressor (optional but recommended)
python scripts/03a_pretrain_compressor.py

# Phase 3: Train latent pager
python scripts/03_train_latent_pager.py

# Phase 4: Evaluate on test set
python scripts/04_evaluate.py

# Phase 5: Run ablation studies
python scripts/05_ablations.py

# Phase 6: Generate comparison report
python scripts/06_generate_report.py
```

## Key Takeaways

If you take away only three things from this post, let them be these:

1. **Latent memory is faster and more accurate than text memory, but hallucinates more.** The speed advantage (2.55x) is real and comes from avoiding expensive text generation during chunking. The accuracy advantage (+41% F1) is statistically significant. The hallucination problem (+98%) is serious and needs to be solved before this approach is production ready.

2. **Simpler architectures beat complex ones when data is limited.** Question conditioning, reconstruction loss, deeper aggregators, more soft tokens. Every complexity we added made things worse. The best model was the simplest one with the right hyperparameters.

3. **The boring engineering decisions matter most.** Pooling strategy (+21% F1). Repetition penalty (test F1 from 0.013 to 0.026). Checkpoint selection metric (35% F1 difference). These unglamorous choices determined the outcome more than any architectural innovation.

The future of long context LLMs is not just about making context windows bigger. It is about building better external memory systems that can store, compress, and retrieve information efficiently. Latent paging is one promising direction. Text buffers are another. The best solution will probably combine both.

---

*Part of my 2026 series on LLM systems research. Full code and results: [github.com/dylanler/rlm-experiment-claude](https://github.com/dylanler/rlm-experiment-claude)*
