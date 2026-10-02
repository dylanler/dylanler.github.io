#!/usr/bin/env python3
"""Jev (Typesafe) side of the world-map experiment — text-only counterpart to Clef."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import requests

ROOT = Path("/workspace/clef-world-map")
TOKEN = os.environ.get("TYPESAFE_API_KEY")
if not TOKEN:
    print("ERROR: TYPESAFE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

API_URL = "https://api.typesafe.ai/v1/systemone"
PHOTO_SOURCE = (
    "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=640&q=70"
)
PHOTO_DESC = "Eiffel Tower, Paris (Unsplash photo-1511739001486-6bfe10ce785f)"
# Text-only caption of the same photo — landmark name implies France (honest limitation)
CAPTION = (
    "Outdoor photograph of the Eiffel Tower in Paris, iron lattice tower against sky, "
    "typical tourist landmark photo."
)

COUNTRIES = [c.strip() for c in (ROOT / "countries.txt").read_text().splitlines() if c.strip()]

# Reuse alias map from run_experiment.py for Natural Earth joins
NAME_ALIASES = {
    "United States": ["United States of America", "United States"],
    "United Kingdom": ["United Kingdom", "United Kingdom of Great Britain and Northern Ireland"],
    "Russia": ["Russia", "Russian Federation"],
    "South Korea": ["South Korea", "Korea", "Republic of Korea"],
    "North Korea": ["North Korea", "Dem. Rep. Korea", "Democratic People's Republic of Korea"],
    "Czechia": ["Czechia", "Czech Republic"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"],
    "Congo": ["Congo", "Republic of the Congo", "Congo (Brazzaville)"],
    "Democratic Republic of the Congo": [
        "Democratic Republic of the Congo",
        "Dem. Rep. Congo",
        "DR Congo",
        "Congo (Kinshasa)",
    ],
    "Tanzania": ["Tanzania", "United Republic of Tanzania"],
    "Vietnam": ["Vietnam", "Viet Nam"],
    "Syria": ["Syria", "Syrian Arab Republic"],
    "Iran": ["Iran", "Iran (Islamic Republic of)"],
    "Bolivia": ["Bolivia", "Bolivia (Plurinational State of)"],
    "Venezuela": ["Venezuela", "Venezuela (Bolivarian Republic of)"],
    "Moldova": ["Moldova", "Republic of Moldova"],
    "Laos": ["Laos", "Lao PDR", "Lao People's Democratic Republic"],
    "Brunei": ["Brunei", "Brunei Darussalam"],
    "Eswatini": ["Eswatini", "Swaziland"],
    "North Macedonia": ["North Macedonia", "Macedonia"],
    "Palestine": ["Palestine", "Palestine, State of"],
    "Micronesia": ["Micronesia", "Federated States of Micronesia"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde"],
    "Timor-Leste": ["Timor-Leste", "East Timor"],
    "Bahamas": ["Bahamas", "The Bahamas"],
    "Gambia": ["Gambia", "The Gambia"],
    "Vatican City": ["Vatican City", "Vatican", "Holy See"],
    "Turkey": ["Turkey", "Türkiye", "Turkiye"],
    "Myanmar": ["Myanmar", "Burma"],
    "Bosnia and Herzegovina": ["Bosnia and Herz.", "Bosnia and Herzegovina"],
    "Dominican Republic": ["Dominican Rep.", "Dominican Republic"],
    "Central African Republic": ["Central African Rep.", "Central African Republic"],
    "Equatorial Guinea": ["Eq. Guinea", "Equatorial Guinea"],
    "South Sudan": ["S. Sudan", "South Sudan"],
    "Solomon Islands": ["Solomon Is.", "Solomon Islands"],
    "United Arab Emirates": ["United Arab Emirates", "UAE"],
    "Saint Kitts and Nevis": ["St. Kitts and Nevis", "Saint Kitts and Nevis"],
    "Saint Vincent and the Grenadines": [
        "St. Vin. and Gren.",
        "Saint Vincent and the Grenadines",
    ],
    "Sao Tome and Principe": ["São Tomé and Principe", "Sao Tome and Principe"],
    "Antigua and Barbuda": ["Antigua and Barb.", "Antigua and Barbuda"],
}


def scrub_secrets(obj):
    """Recursively drop any keys that look like secrets before writing JSON."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(s in kl for s in ("api_key", "apikey", "authorization", "bearer", "token", "secret")):
                out[k] = "<scrubbed>"
            else:
                out[k] = scrub_secrets(v)
        return out
    if isinstance(obj, list):
        return [scrub_secrets(x) for x in obj]
    if isinstance(obj, str) and obj.startswith("apikey_"):
        return "<scrubbed>"
    return obj


def call_jev(model_name: str) -> dict:
    criteria = {c: f"The photo was taken in {c}" for c in COUNTRIES}
    payload = {
        "model": model_name,
        "state": CAPTION,
        "questions": {
            "country": {
                "type": "choice",
                "instructions": (
                    "Which country was this photograph taken in? "
                    "Base the answer on the landmark and scene described in the state."
                ),
                "criteria": criteria,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    print(f"POST systemone model={model_name} countries={len(COUNTRIES)} state_chars={len(CAPTION)}")
    r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    print(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception:
        print("Non-JSON response body length", len(r.content))
        return {"_http_status": r.status_code, "_error": "non-json", "_body_preview": r.text[:500]}
    data["_http_status"] = r.status_code
    return data


def extract_probs(resp: dict) -> dict | None:
    answers = resp.get("answers") or {}
    country = answers.get("country") or {}
    probs = country.get("probabilities")
    if not isinstance(probs, dict):
        return None
    return {k: float(v) for k, v in probs.items()}


def top_n(probs: dict, n=20):
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]


def render_choropleth(probs: dict, out_png: Path, title: str):
    gdf = gpd.read_file(ROOT / "ne_countries.geojson")
    name_col = "NAME" if "NAME" in gdf.columns else "ADMIN"
    admin_col = "ADMIN" if "ADMIN" in gdf.columns else name_col

    ne_to_prob = {}
    for option, p in probs.items():
        aliases = NAME_ALIASES.get(option, [option])
        for a in aliases:
            ne_to_prob[a.lower()] = p
        ne_to_prob[option.lower()] = p

    def lookup(row):
        for cand in (row.get(name_col), row.get(admin_col)):
            if cand and str(cand).lower() in ne_to_prob:
                return ne_to_prob[str(cand).lower()]
        return 0.0

    gdf = gdf.copy()
    gdf["jev_prob"] = gdf.apply(lookup, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=120)
    gdf.plot(ax=ax, color="#e8e8e8", edgecolor="#888888", linewidth=0.3)
    vmax = max(probs.values()) if probs else 1.0
    cmap = plt.cm.YlOrRd
    nonempty = gdf[gdf["jev_prob"] > 0]
    if len(nonempty):
        nonempty.plot(
            ax=ax,
            column="jev_prob",
            cmap=cmap,
            edgecolor="#444444",
            linewidth=0.4,
            vmin=0,
            vmax=vmax,
            legend=False,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Jev P(country | caption)")
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png} ({out_png.stat().st_size} bytes)")
    matched = int((gdf["jev_prob"] > 0).sum())
    print(f"GeoJSON countries with matched probs: {matched}")


def main():
    print(f"Photo (reference): {PHOTO_DESC}")
    print(f"Source URL: {PHOTO_SOURCE}")
    print(f"Text caption (Jev state): {CAPTION}")
    print(f"NOTE: Jev is text-only; landmark name in caption implies France.")

    # --- Primary: jev-latest ---
    jev_resp = call_jev("jev-latest")
    scrubbed = scrub_secrets(jev_resp)
    (ROOT / "jev_raw.json").write_text(json.dumps(scrubbed, indent=2)[:500000])

    status = jev_resp.get("_http_status")
    resolved_model = jev_resp.get("model")
    usage = jev_resp.get("usage")
    print(f"Resolved model: {resolved_model}")
    print(f"Usage: {usage}")
    if status and status >= 400:
        print("ERROR response:", json.dumps(scrubbed, indent=2)[:3000])
        sys.exit(2)

    probs = extract_probs(jev_resp)
    if not probs:
        print("FAILED to extract probabilities from Jev response.")
        print("Top-level keys:", list(jev_resp.keys()))
        print("answers sample:", json.dumps(jev_resp.get("answers"), indent=2)[:2000])
        sys.exit(2)

    country_ans = (jev_resp.get("answers") or {}).get("country") or {}
    top20 = top_n(probs, 20)
    top20_obj = [{"country": c, "probability": p} for c, p in top20]
    out_json = {
        "model_requested": "jev-latest",
        "model": resolved_model,
        "api": "https://api.typesafe.ai/v1/systemone",
        "modality": "text-only",
        "limitation": (
            "Jev cannot see images. State is a short factual caption of the same Unsplash "
            "Eiffel Tower photo used for Clef. Naming the landmark makes France nearly "
            "trivial for a text model; this is an honest text-only counterpart, not a "
            "vision comparison."
        ),
        "caption": CAPTION,
        "photo": PHOTO_DESC,
        "photo_source_url": PHOTO_SOURCE,
        "choice": country_ans.get("choice"),
        "confidence": country_ans.get("confidence"),
        "usage": usage,
        "top20": top20_obj,
        "n_options": len(probs),
    }
    (ROOT / "jev_probs.json").write_text(json.dumps(out_json, indent=2))
    print("Top 5 (jev-latest):")
    for c, p in top20[:5]:
        print(f"  {c}: {p:.6f}")

    render_choropleth(
        probs,
        ROOT / "jev_world_map.png",
        "Jev (text-only) country probabilities — Eiffel Tower caption",
    )

    # --- Optional: jev-preview (for comparison; primary outputs stay on latest) ---
    preview_note = None
    preview_top5 = None
    preview_usage = None
    preview_model = None
    try:
        preview_resp = call_jev("jev-preview")
        preview_status = preview_resp.get("_http_status")
        preview_probs = extract_probs(preview_resp)
        preview_model = preview_resp.get("model")
        preview_usage = preview_resp.get("usage")
        if preview_status and preview_status >= 400 or not preview_probs:
            preview_note = (
                f"jev-preview skipped/failed (HTTP {preview_status}). "
                f"detail={json.dumps(scrub_secrets(preview_resp))[:500]}"
            )
            print(preview_note)
        else:
            preview_top = top_n(preview_probs, 5)
            preview_top5 = [{"country": c, "probability": p} for c, p in preview_top]
            print(f"jev-preview resolved={preview_model} usage={preview_usage}")
            print("Top 5 (jev-preview):")
            for c, p in preview_top:
                print(f"  {c}: {p:.6f}")
            # Save preview raw alongside but do not overwrite primary map
            (ROOT / "jev_preview_raw.json").write_text(
                json.dumps(scrub_secrets(preview_resp), indent=2)[:500000]
            )
            preview_note = "jev-preview succeeded; raw saved to jev_preview_raw.json (map uses jev-latest)"
    except Exception as e:
        preview_note = f"jev-preview exception: {type(e).__name__}: {e}"
        print(preview_note)

    # Update run_notes.json (preserve existing clef fields)
    notes_path = ROOT / "run_notes.json"
    if notes_path.exists():
        notes = json.loads(notes_path.read_text())
    else:
        notes = {}
    notes["jev_top5"] = top20_obj[:5]
    notes["jev_usage"] = usage
    notes["jev_model"] = resolved_model
    notes["jev_model_requested"] = "jev-latest"
    notes["jev_choice"] = country_ans.get("choice")
    notes["jev_confidence"] = country_ans.get("confidence")
    notes["jev_caption"] = CAPTION
    notes["jev_limitation"] = out_json["limitation"]
    notes["jev_preview_note"] = preview_note
    notes["jev_preview_top5"] = preview_top5
    notes["jev_preview_usage"] = preview_usage
    notes["jev_preview_model"] = preview_model
    notes_path.write_text(json.dumps(notes, indent=2))
    print("Updated run_notes.json")
    print("DONE")


if __name__ == "__main__":
    main()
