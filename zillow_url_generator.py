# zillow_url_generator.py

import os
import requests
import urllib.parse
import json

# -------------------------------
# TOKENS & CONSTANTS
# -------------------------------

# ✅ Do NOT hardcode secrets in code (GitHub blocks pushes)
# These must be set in your environment (or Render later)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "").strip()
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "").strip()

ZILLOW_BASE_URL = "https://www.zillow.com/homes/recently_sold/"
APIFY_ACTOR_URL = "https://api.apify.com/v2/acts/api-empire~zillow-search-scraper/runs"

# 🔒 LOCKED VIEWPORT (DO NOT CHANGE)
VIEWPORT_LAT_DELTA = 0.012
VIEWPORT_LON_DELTA = 0.018
LOCKED_ZOOM = 16


# -------------------------------
# GEO
# -------------------------------
def geocode_address(address: str):
    if not MAPBOX_TOKEN:
        raise RuntimeError("MAPBOX_TOKEN is not set in environment variables")

    url = (
        f"https://api.mapbox.com/geocoding/v5/mapbox.places/"
        f"{urllib.parse.quote(address)}.json"
    )
    r = requests.get(url, params={"access_token": MAPBOX_TOKEN, "limit": 1}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data.get("features"):
        raise ValueError("Geocoding failed")
    lon, lat = data["features"][0]["center"]
    return lat, lon


def build_locked_viewport(lat: float, lon: float):
    return {
        "west": lon - VIEWPORT_LON_DELTA,
        "east": lon + VIEWPORT_LON_DELTA,
        "south": lat - VIEWPORT_LAT_DELTA,
        "north": lat + VIEWPORT_LAT_DELTA,
    }


def _months_to_doz_value(sold_months: int) -> str:
    """
    Zillow 'doz' filter value.
    Keep the existing behavior as default (6m). Only allow 6 or 12.
    """
    if sold_months == 6:
        return "6m"
    if sold_months == 12:
        return "12m"
    raise ValueError("sold_months must be 6 or 12")



# -------------------------------
# ZILLOW URL GENERATOR
# -------------------------------
def generate_zillow_url(address, beds, baths, sqft, year_built, property_type, sold_months: int = 6):
    # Normalize user property type inputs to Zillow filter keys
    pt = (property_type or "").strip().lower()
    alias = {
        # single family
        "sf": "sf",
        "sfr": "sf",
        "single": "sf",
        "single_family": "sf",
        "single-family": "sf",
        # multifamily
        "mf": "mf",
        "multi": "mf",
        "multifamily": "mf",
        "multi_family": "mf",
        # condo
        "c": "con",
        "condo": "con",
        "condominium": "con",
        # townhouse
        "th": "tow",
        "townhouse": "tow",
        "townhome": "tow",
        "town house": "tow",
    }
    pt = alias.get(pt, pt)

    lat, lon = geocode_address(address)
    bounds = build_locked_viewport(lat, lon)

    sqft_min = int(sqft * 0.85)
    sqft_max = int(sqft * 1.15)
    year_min = year_built - 15
    year_max = year_built + 15

    # BED LOGIC (unchanged)
    if pt == "mf" and beds in (6, 7):
        beds_min = 4
        beds_max = beds + 1
    else:
        beds_min = beds - 1
        beds_max = beds + 1

    home_types = {
        "sf": {"value": False},
        "mf": {"value": False},
        "tow": {"value": False},
        "con": {"value": False},
        "apa": {"value": False},
        "apco": {"value": False},
        "manu": {"value": False},
        "land": {"value": False},
    }

    if pt in home_types:
        home_types[pt]["value"] = True
    else:
        raise ValueError("Property type must be sf, mf, c (condo), or th (townhouse)")

    search_state = {
        "pagination": {},
        "isMapVisible": True,
        "isListVisible": True,
        "usersSearchTerm": address,
        "mapBounds": bounds,
        "mapZoom": LOCKED_ZOOM,
        "filterState": {
            "rs": {"value": True},
            "doz": {"value": _months_to_doz_value(sold_months)},  # ✅ only change: 6m vs 12m
            "beds": {"min": beds_min, "max": beds_max},
            "baths": {"min": baths - 1, "max": baths + 1},
            "sqft": {"min": sqft_min, "max": sqft_max},
            "built": {"min": year_min, "max": year_max},
            "sort": {"value": "globalrelevanceex"},
            "fsba": {"value": False},
            "fsbo": {"value": False},
            "nc": {"value": False},
            "cmsn": {"value": False},
            "auc": {"value": False},
            "fore": {"value": False},
            **home_types,
        },
    }

    encoded = urllib.parse.quote(json.dumps(search_state))
    return f"{ZILLOW_BASE_URL}?searchQueryState={encoded}"


# -------------------------------
# ✅ SUBJECT WRAPPER (UNCHANGED DEFAULTS)
# -------------------------------
def generate_zillow_url_from_subject(subject, sold_months: int = 6):
    return generate_zillow_url(
        subject["address"],
        subject["beds"],
        subject["baths"],
        subject["sqft"],
        subject["year_built"],
        subject["property_type"],
        sold_months=sold_months,
    )


# -------------------------------
# APIFY RUNNER
# -------------------------------
def run_apify_scraper(zillow_url: str):
    payload = {"searchUrls": [{"url": zillow_url}]}
    r = requests.post(
        APIFY_ACTOR_URL,
        params={"token": APIFY_TOKEN},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    run_id = data["data"]["id"]

    print("\nApify run started successfully.")
    print(f"Run ID: {run_id}")
    print(f"Dashboard: https://console.apify.com/actors/runs/{run_id}")


# -------------------------------
# CLI ENTRY (DEFAULTS UNCHANGED)
# -------------------------------
if __name__ == "__main__":
    address = input("Address: ").strip()
    beds = int(input("Bedrooms: "))
    baths = float(input("Bathrooms: "))
    sqft = int(input("Square Feet: "))
    year = int(input("Year Built: "))

    # Added: condo + townhouse options
    prop_type = input("Property Type (sf / mf / c / th): ").strip().lower()

    zillow_url = generate_zillow_url(address, beds, baths, sqft, year, prop_type, sold_months=6)
    print("\nGenerated Zillow URL:\n")
    print(zillow_url)

    print("\nStarting Apify scrape...")
    run_apify_scraper(zillow_url)

