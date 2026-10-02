#!/usr/bin/env python3
"""Clef world-map experiment — reconstruction (not an official Cloudflare demo)."""
import base64
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
ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
if not ACCOUNT:
    print("ERROR: CLOUDFLARE_ACCOUNT_ID not set", file=sys.stderr)
    sys.exit(1)
TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
if not TOKEN:
    print("ERROR: CLOUDFLARE_AUTH_TOKEN not set", file=sys.stderr)
    sys.exit(1)

PHOTO_SOURCE = (
    "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=640&q=70"
)
PHOTO_DESC = "Eiffel Tower, Paris (Unsplash photo-1511739001486-6bfe10ce785f)"

# Curated UN-ish English short names (keys for choice criteria)
COUNTRIES = [c.strip() for c in (ROOT / "countries.txt").read_text().splitlines() if c.strip()]

# Map Clef option keys -> Natural Earth NAME / ADMIN aliases for joining
# Natural Earth uses various naming conventions
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


def b64_jpeg(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def call_clef(model_path: str, model_name: str, image_b64: str) -> dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{model_path}"
    criteria = {c: f"The photo was taken in {c}" for c in COUNTRIES}
    payload = {
        "model": model_name,
        "state": (
            "Outdoor landmark / street-view style photograph. "
            "Identify which country the photo was taken in."
        ),
        "images": [image_b64],
        "questions": {
            "country": {
                "type": "choice",
                "instructions": "Which country is this photo in?",
                "criteria": criteria,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    # Do not log token
    print(f"POST {model_path} model={model_name} countries={len(COUNTRIES)} image_bytes≈{len(image_b64)//4*3}")
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    print(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception:
        print("Non-JSON response body length", len(r.content))
        return {"_http_status": r.status_code, "_error": "non-json", "_body_preview": r.text[:500]}
    data["_http_status"] = r.status_code
    return data


def extract_probs(resp: dict) -> dict | None:
    """Return {country: probability} from live API response only."""
    # Workers AI often wraps as {"success": true, "result": {...}}
    result = resp.get("result", resp)
    answers = result.get("answers") or {}
    country = answers.get("country") or {}
    probs = country.get("probabilities")
    if not isinstance(probs, dict):
        return None
    return {k: float(v) for k, v in probs.items()}


def top_n(probs: dict, n=20):
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]


def render_choropleth(probs: dict, out_png: Path, title: str):
    gdf = gpd.read_file(ROOT / "ne_countries.geojson")
    # Prefer NAME, fall back to ADMIN
    name_col = "NAME" if "NAME" in gdf.columns else "ADMIN"
    admin_col = "ADMIN" if "ADMIN" in gdf.columns else name_col

    # Build lookup from Natural Earth names -> probability
    # First invert aliases
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
    gdf["clef_prob"] = gdf.apply(lookup, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), dpi=120)
    # Light gray for zero / missing
    gdf.plot(ax=ax, color="#e8e8e8", edgecolor="#888888", linewidth=0.3)
    # Color non-zero
    vmax = max(probs.values()) if probs else 1.0
    cmap = plt.cm.YlOrRd
    nonempty = gdf[gdf["clef_prob"] > 0]
    if len(nonempty):
        nonempty.plot(
            ax=ax,
            column="clef_prob",
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
    cbar.set_label("Clef P(country | photo)")
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png} ({out_png.stat().st_size} bytes)")
    matched = int((gdf["clef_prob"] > 0).sum())
    print(f"GeoJSON countries with matched probs: {matched}")


def main():
    img_path = ROOT / "input.jpg"
    if not img_path.exists():
        print("Missing input.jpg", file=sys.stderr)
        sys.exit(1)
    image_b64 = b64_jpeg(img_path)
    print(f"Photo: {PHOTO_DESC}")
    print(f"Source URL: {PHOTO_SOURCE}")
    print(f"input.jpg size: {img_path.stat().st_size} bytes")

    # --- Clef ---
    clef_resp = call_clef("@cf/cloudflare/clef", "clef", image_b64)
    (ROOT / "clef_raw.json").write_text(json.dumps(clef_resp, indent=2)[:500000])
    # Strip any accidental secret fields before saving summary
    print("Clef success field:", clef_resp.get("success"), "errors:", clef_resp.get("errors"))
    usage = (clef_resp.get("result") or clef_resp).get("usage")
    print("Clef usage:", usage)

    probs = extract_probs(clef_resp)
    if not probs:
        print("FAILED to extract probabilities from Clef response.")
        print("Top-level keys:", list(clef_resp.keys()))
        result = clef_resp.get("result")
        if isinstance(result, dict):
            print("result keys:", list(result.keys()))
            print("answers sample:", json.dumps(result.get("answers"), indent=2)[:2000])
        else:
            print("body preview:", json.dumps(clef_resp, indent=2)[:2000])
        sys.exit(2)

    top20 = top_n(probs, 20)
    top20_obj = [{"country": c, "probability": p} for c, p in top20]
    out_json = {
        "model": "clef",
        "model_path": "@cf/cloudflare/clef",
        "photo": PHOTO_DESC,
        "photo_source_url": PHOTO_SOURCE,
        "choice": (clef_resp.get("result") or clef_resp).get("answers", {}).get("country", {}).get("choice"),
        "confidence": (clef_resp.get("result") or clef_resp).get("answers", {}).get("country", {}).get("confidence"),
        "usage": usage,
        "top20": top20_obj,
        "n_options": len(probs),
    }
    (ROOT / "clef_probs.json").write_text(json.dumps(out_json, indent=2))
    print("Top 5:")
    for c, p in top20[:5]:
        print(f"  {c}: {p:.6f}")

    render_choropleth(
        probs,
        ROOT / "clef_world_map.png",
        "Clef country probabilities — Eiffel Tower photo (reconstruction)",
    )

    # --- Clef-flash ---
    flash_resp = call_clef("@cf/cloudflare/clef-flash", "clef-flash", image_b64)
    (ROOT / "clef_flash_raw.json").write_text(json.dumps(flash_resp, indent=2)[:500000])
    flash_status = flash_resp.get("_http_status")
    print("clef-flash HTTP:", flash_status, "success:", flash_resp.get("success"), "errors:", flash_resp.get("errors"))
    flash_probs = extract_probs(flash_resp)
    flash_note = None
    if flash_status == 422 or not flash_probs:
        flash_note = f"clef-flash skipped/failed (HTTP {flash_status}). errors={flash_resp.get('errors')}"
        print(flash_note)
    else:
        flash_top = top_n(flash_probs, 20)
        flash_out = {
            "model": "clef-flash",
            "model_path": "@cf/cloudflare/clef-flash",
            "photo": PHOTO_DESC,
            "photo_source_url": PHOTO_SOURCE,
            "usage": (flash_resp.get("result") or flash_resp).get("usage"),
            "top20": [{"country": c, "probability": p} for c, p in flash_top],
        }
        (ROOT / "clef_flash_probs.json").write_text(json.dumps(flash_out, indent=2))
        print("clef-flash top 5:")
        for c, p in flash_top[:5]:
            print(f"  {c}: {p:.6f}")
        render_choropleth(
            flash_probs,
            ROOT / "clef_flash_world_map.png",
            "Clef-flash country probabilities — Eiffel Tower photo",
        )

    # Persist notes for README
    notes = {
        "photo": PHOTO_DESC,
        "photo_source_url": PHOTO_SOURCE,
        "clef_top5": top20_obj[:5],
        "clef_usage": usage,
        "flash_note": flash_note,
        "flash_usage": ((flash_resp.get("result") or flash_resp).get("usage") if flash_probs else None),
    }
    (ROOT / "run_notes.json").write_text(json.dumps(notes, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
