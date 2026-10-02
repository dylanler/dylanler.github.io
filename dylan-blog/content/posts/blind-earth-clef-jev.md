+++
title = 'Blind Earth with Clef, Clef-flash, and Jev'
date = 2026-10-02T01:20:00-07:00
draft = false
tags = ["AI", "Cloudflare", "decision-models", "Jev", "geography", "System One"]
+++

Henry’s [How Does A Blind Model See The Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) asks a language model, cell by cell, whether a lat/lon is over land or water, then paints the answers on an equirectangular grid. No images go in. Whatever structure appears in the map is whatever geographic prior the model already carries.

This post adapts that recipe for **System One** decision models — Cloudflare **Clef** / **Clef-flash** and TypeSafe **Jev** — using a binary `choice` (Land vs Water) instead of free-form generation. Code, grids, and the three hard B&W posters live in [`dylanler/blind-earth-clef-jev`](https://github.com/dylanler/blind-earth-clef-jev).

> An earlier draft on this site used a country-list choropleth on an Eiffel Tower photo. That was the wrong experiment; see the [supersede note](/posts/clef-world-map-decision-model/).

## Method

**Grid.** Latitudes `-89 … +89` and longitudes `-179 … +179`, both step **2°** → **90 × 180 = 16,200** points. Batch size **64** → **254** requests per model. Equirectangular, north-up.

**Question.** For each cell, one System One `choice`:

> Is the location at *φ°N/S, λ°E/W* over land or over water?

Criteria: **Land** = continents, islands, ice, snow; **Water** = oceans, seas, other open water. Shared `state` tells the model it is answering geographic land/water questions about Earth coordinates.

**Hard map.** White where `P(Land) > 0.5`, black otherwise. Soft greyscales (raw `P(Land)`) are in the repo under `maps/`.

**APIs.** Clef and Clef-flash via Cloudflare Workers AI (`CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_AUTH_TOKEN`). Jev via TypeSafe (`TYPESAFE_API_KEY`, `model: "jev-latest"`). Credentials stay in the environment; nothing secret is committed.

## Results

| Model | Wall (8 workers) | Hard land fraction | Mean P(Land) |
|---|---:|---:|---:|
| Clef | ~48.5 s | **0.383** | 0.416 |
| Clef-flash | ~27.7 s | **0.511** | 0.487 |
| Jev | ~5.0 s | **0.487** | 0.484 |

Earth’s true land fraction is about **0.29**. All three over-predict land on this grid — especially near the poles, where ice/snow criteria and sparse training signal both pull toward Land.

### Clef

Americas, Africa, Eurasia, Australia, and Antarctica are all readable. Some Pacific salt-and-pepper. Lowest hard land fraction of the three (~38%) — closest to the real ~29%, though still high.

![Clef blind Earth — hard B/W, white=Land](/images/blind_earth_clef.png)

### Clef-flash

Same recipe, noisier coasts. Land sits in roughly the right longitudinal bands, but speckles fill more of the oceans. Highest hard land fraction (~51%).

![Clef-flash blind Earth — hard B/W, white=Land](/images/blind_earth_clef_flash.png)

### Jev

Continents are recognizable; Afro-Eurasia tends to merge; Antarctica shows up as a thick southern white band. More false-land speckles in the Pacific than Clef. Hard land ~49%, mean P(Land) ~0.48.

![Jev blind Earth — hard B/W, white=Land](/images/blind_earth_jev.png)

## What this shows (and does not)

- Decision-model `choice` is enough to turn a geographic prior into a map without any image input.
- Clef’s hard map is the cleanest of the three here; flash and Jev are more land-happy and noisier over open ocean.
- This does **not** measure vision quality — none of these calls saw pixels.
- This does **not** claim an official Cloudflare or TypeSafe “world map” demo. It is a reconstruction of Henry’s blind-Earth idea on System One APIs.
- Equirectangular area distortion and the ice/snow “Land” criterion both inflate land fraction relative to a true surface-area number.

## Reproduce

```bash
git clone https://github.com/dylanler/blind-earth-clef-jev
cd blind-earth-clef-jev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_AUTH_TOKEN=...
export TYPESAFE_API_KEY=...

python3 run_blind_earth.py --model all --workers 8
```

Full timing, token usage, and resume notes: [`run_log.md`](https://github.com/dylanler/blind-earth-clef-jev/blob/main/run_log.md) in the how-to repo.

## Links

- Recipe: [How Does A Blind Model See The Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) (Henry / outsidetext)
- How-to + artifacts: [github.com/dylanler/blind-earth-clef-jev](https://github.com/dylanler/blind-earth-clef-jev)
