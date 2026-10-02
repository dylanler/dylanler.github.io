+++
title = 'Clef + Jev World Map: Decision Models on a Landmark Photo'
date = 2026-10-02T01:00:00-07:00
draft = false
tags = ["AI", "Cloudflare", "decision-models", "Jev", "vision"]
+++

This is a **reconstruction**, not an official Cloudflare demo. There is no published “Clef world-map” recipe. I wired Workers AI Clef to a country-choice question, painted the returned probabilities on a Natural Earth choropleth, and ran the same choice question through TypeSafe’s text-only Jev with a short caption of the same photo.

The question is simple:

> Given one outdoor landmark photo and ~195 country options, where does a vision decision model put its mass—and what happens when a text-only model gets a caption that already names the landmark?

## Hypothesis

Clef should concentrate probability on France for an unambiguous Eiffel Tower photo. Clef-flash should agree on the mode but with a flatter distribution. Jev, receiving a caption that names the Eiffel Tower and Paris, should collapse almost entirely onto France. That last result is expected and **not** a fair vision comparison—it is an honest text-only counterpart.

## Method

**Photo.** Unsplash Eiffel Tower, Paris: [photo-1511739001486-6bfe10ce785f](https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=640&q=70). Local copy resized for the API.

![Input: Eiffel Tower, Paris](/images/clef-world-map-input-eiffel.jpg)

**Clef (vision).** Call `@cf/cloudflare/clef` with `model: "clef"`, a short outdoor-landmark `state`, the JPEG as a base64 data URI, and one `choice` question—*Which country is this photo in?*—with ~195 English short country names as criteria. Parse `result.answers.country.probabilities` only from the live response. Render a world choropleth. Repeat against `@cf/cloudflare/clef-flash`.

**Jev (text-only).** Same ~195-country `choice` against TypeSafe `jev-latest` (resolved `jev-1.13.0`). No image. `state` is a short factual caption of the same photo:

> Outdoor photograph of the Eiffel Tower in Paris, iron lattice tower against sky, typical tourist landmark photo.

Reproduction scripts live in [`experiment-tools/clef_world_map/`](https://github.com/dylanler/dylanler.github.io/tree/main/experiment-tools/clef_world_map). Credentials stay in the environment; nothing secret is committed.

## Results

| Model | Modality | Top country | Probability | Confidence |
|---|---|---|---:|---:|
| Clef | vision (JPEG) | France | 0.852 | 0.7246 |
| Clef-flash | vision (JPEG) | France | 0.6359 | — |
| Jev `jev-1.13.0` | text (caption) | France | 1.0 | 1.0 |

### Clef

France dominates at **0.852** with confidence **0.7246**. The next countries sit near the noise floor (~0.0015). The map shows a clear France hotspot and a near-uniform wash elsewhere.

![Clef world-map choropleth](/images/clef_world_map.png)

### Clef-flash

Same mode—France—but softer: **0.6359**. Runner-ups are still tiny (~0.003). The choropleth looks similar with a less saturated peak.

![Clef-flash world-map choropleth](/images/clef_flash_world_map.png)

### Jev

Requested `jev-latest`, resolved **`jev-1.13.0`**. Choice **France** at probability **1.0** / confidence **1.0**. That is the right answer given a caption that already says “Eiffel Tower in Paris.” It is also nearly trivial for a text model. Treat the map as a text baseline, not evidence that Jev “sees” the tower.

![Jev world-map choropleth](/images/jev_world_map.png)

## What this does not prove

- It does not claim Cloudflare ships an official world-map demo.
- It does not compare vision quality between Clef and Jev; Jev never saw pixels.
- It does not stress ambiguous landmarks, multi-country scenes, or adversarial captions.
- It does not invent probabilities—the numbers above come from live API responses on this photo and this criteria list.

## Takeaway

Decision-model `choice` over a large country list is a clean way to turn model uncertainty into a map. On an easy landmark, Clef and Clef-flash both pick France with meaningful but different peak mass. Jev’s France=1.0 is the expected text-only outcome when the caption does the hard work. The interesting next experiment is the hard case: withhold the landmark name, or use a photo where the country is not obvious from a one-line description.
