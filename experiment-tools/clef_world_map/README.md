# Clef + Jev world-map experiment (reconstruction)

This is a **reconstruction**, not an official Cloudflare demo. There is no published “Clef world-map” recipe; this folder wires Workers AI Clef to a country-choice question and paints the returned probabilities on a Natural Earth choropleth. Jev provides a text-only counterpart using a short caption of the same photo.

## What it does

1. Takes one outdoor landmark JPEG.
2. Calls Workers AI `@cf/cloudflare/clef` with `model: "clef"`, a short `state`, the image as a base64 data URI, and one `choice` question: *Which country is this photo in?* Criteria keys are ~195 English short country names (list in `countries.txt`).
3. Parses `result.answers.country.probabilities` from the live API response (no invented values).
4. Renders a world choropleth PNG colored by those probabilities and writes a top-20 JSON summary.
5. Repeats the same payload against `@cf/cloudflare/clef-flash` (`model: "clef-flash"`).
6. Runs the same choice question against TypeSafe Jev (`jev-latest`) with a short factual caption instead of pixels.

## Photo used

- **Landmark:** Eiffel Tower, Paris (unambiguous country: France)
- **Source URL:** https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=640&q=70

## How to re-run (Clef)

```bash
export CLOUDFLARE_ACCOUNT_ID=...   # your Workers AI account id
export CLOUDFLARE_AUTH_TOKEN=...   # never commit
python3 -m venv .venv && source .venv/bin/activate
pip install pillow requests geopandas matplotlib
# Place a Natural Earth 110m admin-0 GeoJSON as ne_countries.geojson
python3 run_experiment.py
```

API endpoint shape:

```
POST https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef
Authorization: Bearer $CLOUDFLARE_AUTH_TOKEN
Content-Type: application/json

{
  "model": "clef",
  "state": "Outdoor landmark / street-view style photograph. Identify which country the photo was taken in.",
  "images": ["data:image/jpeg;base64,..."],
  "questions": {
    "country": {
      "type": "choice",
      "instructions": "Which country is this photo in?",
      "criteria": { "<Country Name>": "The photo was taken in <Country Name>", ... }
    }
  }
}
```

## How to re-run (Jev)

Jev is **text-only** (no images). `state` is a short factual caption of the same Unsplash Eiffel Tower photo:

> Outdoor photograph of the Eiffel Tower in Paris, iron lattice tower against sky, typical tourist landmark photo.

**Limitation:** naming the landmark makes France nearly trivial for a text model. This is an honest text-only counterpart to the Clef vision run, **not** a fair vision comparison.

```bash
export TYPESAFE_API_KEY=...   # from console.typesafe.ai; never commit
python3 run_jev_experiment.py
```

## Notes

- Choice criteria can be large (here ~195 options; keep under 255).
- Clef requires the image embedded as a base64 data URI; remote image URLs are not accepted.
- Auth tokens and account IDs are read from the environment and never printed or committed.
- Do not commit `.venv`, raw API envelopes that may contain secrets, or credentials.
