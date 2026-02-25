+++
title = 'What If LLMs Remembered in Vectors Instead of Words?'
date = 2026-02-25T14:00:00-08:00
draft = false
tags = ["AI", "LLM", "memory", "latent-space", "long-context", "experiment", "transformers"]
+++

What happens when you give a language model a 60,000 token document and ask it a question about paragraph 47?

It forgets. Or worse, it makes something up.

The standard fix is to chop the document into chunks, summarize each chunk into text, glue the summaries together, and feed that back to the model. This is the Recursive Language Model approach and it works okay. But every time you squeeze information through the vocabulary bottleneck (turning hidden states into words), you lose something. Nuance. Uncertainty. The subtle distributional signals that a transformer builds up internally but can never fully express in tokens.

So I ran an experiment: what if instead of summarizing each chunk into text, we saved the model's raw hidden states as compressed vectors and let a neural network figure out how to use them later?

I called it Latent Pager Memory. The name comes from virtual memory paging in operating systems. Instead of paging text to disk, you page latent states.

## How It Works

The idea is straightforward. Take a long document. Chunk it into 1024 token pieces. Run each chunk through a frozen Qwen3-1.7B and grab the hidden states from four different layers (layers 7, 14, 21, and 27). That gives you a 4×2048 = 8192 dimensional snapshot of what the model "understood" from that chunk.

Then compress that 8192 dimensional vector down to 512 dimensions using a small neural network (a PageCompressor). Store these compressed "page vectors" somewhere.

When a question comes in, take all the page vectors, feed them into a Perceiver style cross attention module (a PageAggregator) that produces 16 soft prompt tokens. Prepend those soft prompt tokens to the question and let the frozen LM generate an answer.

The entire pipeline looks like this:

```
Document → chunk (1024 tokens) → frozen LM forward pass → extract hidden states
→ compress to 512 dim → store page vector → repeat for all chunks
→ aggregate all pages via cross attention → 16 soft prompt tokens
→ prepend to question → frozen LM generates answer
```

The baseline does something different: it runs the LM to generate a text summary for each chunk, concatenates all the summaries, and runs the LM again to generate a final answer. More LM calls, more text generation, more places where information gets lost.

## The Setup

I ran everything on 4x A100 80GB GPUs with Qwen3-1.7B as the base model (frozen throughout). The trainable parts are only the PageCompressor (9.4M params) and PageAggregator (82.2M params), totaling 91.6M trainable parameters sitting on top of a 1.7B frozen model.

The dataset was 2,800 synthetic QA samples built from Wikipedia, arXiv, and news articles. Documents ranged from 8K to 65K tokens. Two task types: single fact extraction ("Where was X born?") and multi hop reasoning ("Which of X and Y was published first?").

## What Actually Happened

Here is the headline result after three iterations of trying:

| Metric | Text Buffer (Baseline) | Latent Pager | Change |
|---|---|---|---|
| F1 | 0.0182 | **0.0257** | +41% |
| ROUGE-L | 0.0177 | **0.0260** | +47% |
| Hallucination Rate | **0.292** | 0.580 | +98% (bad) |
| Avg Latency | 19.55s | **7.65s** | 2.55x faster |

All differences were statistically significant (p < 0.001, 10,000 bootstrap iterations).

The latent pager is genuinely better at answering questions (higher F1 and ROUGE-L) and genuinely faster (because it doesn't need to generate text summaries for each chunk). But it hallucinates way more. Almost double the hallucination rate.

This is the central tension of the experiment: the model gets closer to the right answer more often, but when it's wrong, it's wrong with high confidence and fabricated details.

## The Three Iterations (And Why Simpler Won)

I did not get here on the first try. In fact the first two attempts failed outright.

**Version 1** used the initial hyperparameters I picked (mean pooling, 32 soft tokens, 2 aggregator layers). The result: F1 of 0.0136, which is *worse* than the baseline. Not great.

Then I ran ablation studies. I swept across pooling strategies, number of soft tokens, aggregator depth, compression dimension, and extraction layers. The ablations revealed something interesting: three individual settings each independently beat the baseline.

| Setting | F1 | What it beat |
|---|---|---|
| last_token pooling (vs mean) | 0.0231 | Baseline 0.0182 |
| 16 soft tokens (vs 32) | 0.0240 | Baseline 0.0182 |
| 1 aggregator layer (vs 2) | 0.0232 | Baseline 0.0182 |

So the original model was being held back by bad hyperparameters. Not a bad architecture, bad settings.

**Version 2** went too far in the other direction. I added question conditioned aggregation (a bottleneck projection that biases the aggregator based on the question) and a reconstruction auxiliary loss (forcing page vectors to be able to reconstruct original hidden states). Both sounded smart. Both made things worse. Test F1 dropped to 0.0143. The question conditioning added 4.5M extra parameters that overfitted on the small training set. The reconstruction loss pulled the training signal away from what actually matters: answering questions correctly.

**Version 3** was the simplest: just apply the ablation optimal settings (last_token pooling, 16 soft tokens, 1 aggregator layer), use the pretrained compressor, and keep everything else minimal. No question conditioning. No reconstruction loss. This version reached F1 of 0.0257.

The lesson was clear. On a small dataset (2,000 training samples) with a small model (1.7B), every extra parameter you add is a parameter that overfits. Simpler wins.

## The Ablation Findings in Detail

These are the results that actually guided the final design.

### Pooling: Last Token Crushes Mean

| Strategy | F1 | Hallucination |
|---|---|---|
| Mean pooling | 0.0191 | 0.273 |
| **Last token** | **0.0231** | **0.073** |

This was the single biggest lever. Last token pooling gave a 21% F1 boost and a 73% reduction in hallucination. Why? Averaging (mean pooling) dilutes the signal. The last token position in a transformer is where attention has already aggregated the most task relevant information across the sequence. Pulling from that position preserves the model's own internal summary.

### Number of Soft Tokens: 16 is the Sweet Spot

| Tokens | F1 | Hallucination |
|---|---|---|
| 8 | 0.0186 | 0.211 |
| **16** | **0.0240** | 0.271 |
| 32 | 0.0191 | 0.273 |
| 64 | 0.0171 | 0.316 |
| 128 | 0.0163 | 0.261 |

More tokens means more parameters in the aggregator, which means more overfitting. 16 tokens was enough to carry the compressed information. 8 was too few (not enough bandwidth). 32+ started hurting.

### Aggregator Depth: One Layer Is Enough

| Depth | F1 | Hallucination |
|---|---|---|
| **1 layer** | **0.0232** | 0.330 |
| 2 layers | 0.0191 | 0.273 |
| 4 layers | 0.0181 | 0.194 |

With only ~2 chunks per document on average, there just isn't enough page to page interaction to justify deep cross attention. One layer was sufficient. Interestingly, 4 layers had the lowest hallucination (0.194) even though its F1 was worst. Deeper models were more cautious but less accurate.

### Page Dimension: 512 Balances Compression and Quality

| d_page | F1 | Compression Factor |
|---|---|---|
| 128 | 0.0185 | 64x |
| 256 | 0.0153 | 32x |
| **512** | **0.0191** | **16x** |
| 1024 | 0.0161 | 8x |
| 2048 | 0.0179 | 4x |

There was no clean monotonic relationship between dimension and quality. 512 gave the best F1 at 16x compression. More capacity didn't help, probably because the aggregator also grows with d_page and overfits faster.

## The Repetition Penalty Breakthrough

There is a detail that isn't glamorous but made a huge difference. After training version 3 and evaluating it, the test F1 was 0.013. Way worse than the 0.029 validation F1. I looked at the predictions and found the model was generating repetitive loops:

```
The answer is \boxed{Napoleon}. The answer is \boxed{Napoleon}. The answer is
\boxed{Napoleon}. The answer is \boxed{Napoleon}. The answer is \boxed{Napoleon}...
```

The model had learned to output the right content but couldn't stop. Two fixes solved this:

1. Add `repetition_penalty=1.3` to the generate call
2. Post process outputs to truncate at the first repeated sentence

After these changes, test F1 jumped from 0.013 to 0.0257. More than half the final performance came from fixing generation quality, not architecture changes.

## What the Predictions Actually Look Like

Both systems struggle with this task. Qwen3-1.7B is simply too small to reliably answer complex questions about long documents. But the failure modes are different and revealing.

**Question:** "Where did the hijacked plane rammed by Heather Penney crash?"
**Gold answer:** Pennsylvania

The latent pager answered: "The 1980s. The plane crashed in the ocean off of a small island near where Heather Penney was born... Answer: D: Texas"

The baseline answered: "The hijacked plane rammed by Heather Penney crashed into a field in Pennsylvania." (Then repeated this explanation five times.)

The baseline actually got this one right but buried it in repetition. The latent pager fabricated a confident but completely wrong answer with specific details that don't exist.

**Question:** "Which has more species, Festuca or Dactylis?"
**Gold answer:** Festuca

The latent pager answered: "Festuca has more species. 1057 different types of Festuca..." (correct direction, fabricated number)

The baseline answered: "Based on the information provided, the answer is Dactylis..." (wrong, with long wrong reasoning)

Here the latent pager got the right answer but hallucinated supporting evidence. The baseline got it wrong but with lower confidence. This pattern repeated across the dataset: the latent pager was right more often but confabulated details freely.

## Why Hallucination Got Worse

This is the most important question. The whole motivation was that latent states should be more faithful than text summaries, so why does the latent pager hallucinate more?

My best theory: the soft prompt injection creates a "modality gap." The frozen LM was trained on text token embeddings. The soft prompt tokens come from a completely different distribution (the output of a cross attention module over compressed page vectors). The LM doesn't know what to do with these unusual embeddings, so it falls back on its priors, which means generating plausible sounding text that isn't grounded in the actual input.

The text baseline, by contrast, gives the LM text it can actually read and ground its answers in. The summaries are lossy, but at least they're in a format the model understands.

This suggests that the approach might work much better with LoRA tuning of the base model to help it interpret soft prompts. Or with a larger model that has more capacity to bridge the modality gap.

## Five Hypotheses and What Happened

Before running the experiment, I registered five hypotheses:

**H1: Latent pages reduce hallucination by at least 10%.** NOT SUPPORTED. Hallucination went up 98%. The central claim of the experiment was wrong at this scale.

**H2: Multi hop accuracy improves by at least 5 F1 points.** SUPPORTED (weakly). Multi hop F1 went from 0.0155 to 0.0195, a 26% relative improvement. The absolute gain was small but statistically significant and in the right direction.

**H3: Global consistency improves.** INCONCLUSIVE. The dataset didn't include consistency evaluation tasks.

**H4: Information retention scales with d_page.** SUPPORTED. The ablation showed a clear capacity/quality tradeoff, though the relationship was not monotonically increasing.

**H5: Compute cost is at most 1.5x baseline.** SUPPORTED. The latent pager was actually 2.55x faster. This makes sense: the baseline generates text summaries for every chunk (expensive), while the latent pager only does forward passes (cheap) and one final generation.

## What I Would Do Differently

If I were to run this experiment again, here is what I would change.

**Use a bigger model.** Both systems get F1 under 0.03. Qwen3-1.7B just isn't big enough for long document QA. The latent pager's advantage might be more pronounced with a 7B or 13B model that can actually answer the questions.

**Address hallucination directly.** Add a contrastive loss that explicitly penalizes soft prompts that lead to unfaithful generation. Or train a small classifier to score faithfulness and use it for rejection sampling.

**Test on longer documents.** The average document in this dataset was ~1,800 tokens, which is only 2 chunks. The latent pager's aggregation advantage should grow with more chunks. Test on 100K+ token documents where the baseline's recursive summarization would compound errors across many steps.

**LoRA tune the frozen model.** The modality gap between learned soft prompts and the frozen LM's expectations is likely a major source of hallucination. Adding LoRA adapters to help the base model understand the soft prompt distribution could close this gap.

**Use established benchmarks.** The synthetic QA dataset has limitations. NarrativeQA, QuALITY, or SCROLLS would provide better comparison points and make results more meaningful.

## The Bigger Picture

This experiment tested a simple idea: what if LLMs could remember in vectors instead of words? The answer is nuanced. At 1.7B scale with 2,000 training samples, the latent approach is faster and more accurate but less faithful. The speed advantage is real and meaningful (2.55x). The accuracy advantage is statistically significant but small in absolute terms. The hallucination problem is serious and unsolved.

The most surprising takeaway was how much the final result depended on mundane engineering decisions. Pooling strategy. Repetition penalty. Checkpoint selection metric. Number of soft tokens. These "boring" choices mattered more than any architectural innovation.

I think the approach has genuine promise at larger scale. Text summaries are a fundamentally lossy intermediate representation. There should be a point where preserving the continuous state pays off. This experiment didn't quite find that point, but it showed the direction is viable.

The full codebase, data, and results are at [github.com/rlm-exp-claude](https://github.com/dylanler/rlm-exp-claude). The interactive report with charts and ablation visualizations is available as a static site in the repo.

---

*Part of my 2026 series on LLM systems research. Sometimes the best architecture is the simplest one, and the biggest gains come from the smallest fixes.*
