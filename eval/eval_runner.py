# eval/eval_runner.py
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# ------------------------------------------------------------------
# ABSOLUTE FIRST: Load .env and force env vars into the process
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"

def _load_dotenv_force(dotenv_path: Path):
    if not dotenv_path.exists():
        return
    try:
        from dotenv import dotenv_values
        values = dotenv_values(str(dotenv_path))
        for k, v in values.items():
            if k and v is not None:
                os.environ.setdefault(k, str(v).strip())
        return
    except Exception:
        pass
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)

_load_dotenv_force(DOTENV_PATH)

# ------------------------------------------------------------------
# ENV SAFETY SHIMS
# ------------------------------------------------------------------
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
if not os.environ.get("AWS_DEFAULT_REGION") and os.environ.get("AWS_REGION"):
    os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"]

# ------------------------------------------------------------------
# EARLY DIAGNOSTICS: prove which python & torch are being used
# ------------------------------------------------------------------
def _print_runtime_diagnostics():
    print("\n==============================")
    print("RUNTIME DIAGNOSTICS")
    print("==============================")
    print("sys.executable:", sys.executable)
    print("sys.version:", sys.version.replace("\n", " "))
    print("cwd:", os.getcwd())
    print("repo_root:", str(REPO_ROOT))
    print("dotenv:", str(DOTENV_PATH), "exists=", DOTENV_PATH.exists())
    print("PATH head:", os.environ.get("PATH", "")[:200], "...")
    print("==============================\n")

def _torch_sanity_check():
    """
    If torch can't import here, pipeline will fail later anyway.
    Also print where torch is imported from to catch multiple installs.
    """
    try:
        import torch  # noqa
        print("TORCH IMPORT: OK")
        print("torch.__version__:", getattr(torch, "__version__", "unknown"))
        print("torch.__file__:", getattr(torch, "__file__", "unknown"))
        return True
    except Exception as e:
        print("TORCH IMPORT: FAILED")
        print("error:", repr(e))
        return False

_print_runtime_diagnostics()

# If torch import is flaky, fail immediately with clear info
if not _torch_sanity_check():
    print("\nFix: you are running a python/torch combo that cannot load torch DLLs.")
    print("Run THIS EXACT command in the SAME shell you ran eval from:")
    print('  python -c "import torch; print(torch.__version__, torch.__file__)"')
    sys.exit(1)

# ------------------------------------------------------------------
# Now safe to import everything else
# ------------------------------------------------------------------
import pandas as pd

sys.path.append(str(REPO_ROOT))
from orchestrator import run_full_underwrite  # import AFTER torch sanity check

NOT_ENOUGH_TOKEN = "NOT_ENOUGH_USABLE_COMPS"

# ------------------------------------------------------------------
# EXPORT HELPERS (MATCH WEBSITE OUTPUTS)
# ------------------------------------------------------------------
def fmt_money(v):
    try:
        return f"${int(float(v)):,.0f}"
    except Exception:
        return ""

def _extract_arv_value(arv_container):
    v = arv_container
    for _ in range(3):
        if isinstance(v, dict) and "arv" in v:
            v = v.get("arv")
        else:
            break
    return float(v)

def is_not_enough_comps(arv_obj: dict) -> bool:
    if not isinstance(arv_obj, dict):
        return False
    arv_status = str(arv_obj.get("status") or "").lower().strip()
    arv_msg = str(arv_obj.get("message") or "").upper().strip()
    arv_value_direct = arv_obj.get("arv", None)

    return (
        "NOT_ENOUGH_USABLE_COMPS" in arv_msg
        or (arv_value_direct is None and "NOT_ENOUGH" in arv_msg)
        or (arv_value_direct is None and arv_status in ["fail", "failed"])
    )

def get_comps_used(arv_obj: dict, result: dict):
    """
    Match main.py:
      - prefer arv_obj["selected_comps_enriched"]
      - fallback to result["arv"]["selected_comps_enriched"]
      - fallback to arv_obj["selected_comps"]
    """
    comps_out = []
    try:
        if isinstance(arv_obj, dict):
            comps_out = arv_obj.get("selected_comps_enriched") or []
    except Exception:
        comps_out = []

    if not comps_out:
        try:
            arv2 = (result.get("arv") if isinstance(result, dict) else None)
            if isinstance(arv2, dict):
                comps_out = arv2.get("selected_comps_enriched") or []
        except Exception:
            comps_out = []

    if not comps_out:
        try:
            if isinstance(arv_obj, dict):
                comps_out = arv_obj.get("selected_comps") or []
        except Exception:
            comps_out = []

    return comps_out if isinstance(comps_out, list) else []

def compute_mao_like_website(arv: int, rehab: int, subject: dict):
    deal_type = (subject.get("deal_type") or "").lower().strip()
    assignment_fee = float(subject.get("assignment_fee") or 0)

    if deal_type == "rental":
        mao = int(arv * 0.85 - rehab)
    elif deal_type == "flip":
        mao = int(arv * 0.75 - rehab)
    elif deal_type == "wholesale":
        mao = int(arv * 0.75 - rehab - assignment_fee)
    else:
        mao = int(arv * 0.75 - rehab)

    return max(mao, 0)

# ------------------------------------------------------------------
# INPUT HELPERS
# ------------------------------------------------------------------
def parse_money_or_token(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    su = s.upper().strip()
    if "NOT ENOUGH" in su:
        return NOT_ENOUGH_TOKEN
    s = s.replace(",", "").replace("$", "").strip().lower()
    try:
        if s.endswith("k"):
            return float(s[:-1]) * 1000.0
        if s.endswith("m"):
            return float(s[:-1]) * 1_000_000.0
        return float(s)
    except Exception:
        return None

def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default

def safe_beds_int(x, default=0):
    return safe_int(x, default=default)

def normalize_property_type_for_backend(x):
    """
    Must match main.py / zillow_url_generator expectations:
      sf / mf / c / th
    """
    s = (str(x) if x is not None else "").strip().lower()
    if "single" in s or s in ["sf", "sfr"]:
        return "sf"
    if "multi" in s or s in ["mf", "multifamily", "multi_family"]:
        return "mf"
    if "condo" in s or s in ["c", "con"]:
        return "c"
    if "town" in s or s in ["th", "tow", "townhouse", "townhome", "town house"]:
        return "th"
    return "sf"

def extract_model_arv(result_dict):
    if not isinstance(result_dict, dict):
        return None
    arv_section = result_dict.get("arv", {})
    if isinstance(arv_section, dict):
        v = arv_section.get("arv", None)
        try:
            return float(v)
        except Exception:
            if isinstance(v, str) and v.upper().strip() == NOT_ENOUGH_TOKEN:
                return NOT_ENOUGH_TOKEN
            return None
    if isinstance(arv_section, str) and arv_section.upper().strip() == NOT_ENOUGH_TOKEN:
        return NOT_ENOUGH_TOKEN
    try:
        return float(arv_section)
    except Exception:
        return None

def make_logger(prefix: str):
    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{ts} | {prefix} | {msg}")
    return _log

def run_with_timeout(subject: dict, timeout_seconds: int):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_full_underwrite, subject, make_logger(subject.get("address","")[:40]))
        try:
            return fut.result(timeout=timeout_seconds), None, None
        except FuturesTimeout:
            return None, f"Timed out after {timeout_seconds}s (likely model load/inference).", "TIMEOUT"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV test set")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "eval" / "output"), help="Output folder")
    ap.add_argument("--check-env", action="store_true", help="Print whether key env vars are present (no secrets).")
    ap.add_argument("--timeout", type=int, default=240, help="Per-property timeout seconds (default 240)")
    ap.add_argument("--top-comps", type=int, default=3, help="How many top comps to export (default 3, max 3)")
    args = ap.parse_args()

    top_n = int(args.top_comps or 3)
    top_n = max(0, min(top_n, 3))

    if args.check_env:
        keys = [
            "MAPBOX_TOKEN","APIFY_TOKEN",
            "AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY",
            "AWS_REGION","AWS_DEFAULT_REGION",
            "S3_BUCKET_NAME","MODEL_S3_BUCKET",
            "AWS_EC2_METADATA_DISABLED",
        ]
        print("ENV CHECK (True means present):")
        for k in keys:
            print(f"  {k}: {bool(os.getenv(k))}")
        print(f"\nLoaded .env path: {DOTENV_PATH} (exists={DOTENV_PATH.exists()})")
        return

    inp = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    df = df.rename(columns={
        "Example Number": "example_number",
        "Address": "address",
        "# of Bedrooms": "beds",
        "# of Bathrooms": "baths",
        "Square feet": "sqft",
        "Year built": "year_built",
        "# of Units": "units",
        "Property Type ": "property_type",
        "My Results": "my_results",
        "Dottid.AI Results": "dottid_results",
    })

    required_cols = ["address","beds","baths","sqft","year_built","units","property_type","my_results"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns after rename:", missing)
        print("Columns found:", list(df.columns))
        return

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rows_out = []

    total = len(df)
    priced_count = 0
    no_comp_count = 0
    within_5 = 0
    within_10 = 0
    decision_agree = 0
    decision_total = 0

    # Pre-build comp columns so every row has same schema
    def blank_comp_fields():
        fields = {}
        for n in (1, 2, 3):
            fields.update({
                f"comp{n}_zpid": None,
                f"comp{n}_address": None,
                f"comp{n}_sold_price": None,
                f"comp{n}_sold_price_str": None,
                f"comp{n}_distance_miles": None,
                f"comp{n}_beds": None,
                f"comp{n}_baths": None,
                f"comp{n}_sqft": None,
                f"comp{n}_zillow_url": None,
            })
        return fields

    for i, row in df.iterrows():
        idx = i + 1
        address = str(row.get("address", "")).strip()
        print(f"\n[{idx}/{total}] Running: {address}")

        baseline_arv = parse_money_or_token(row.get("my_results", None))

        beds = safe_beds_int(row.get("beds", 0))
        baths = safe_float(row.get("baths", 0))
        sqft = safe_int(row.get("sqft", 0))
        year_built = safe_int(row.get("year_built", 0))
        units = safe_int(row.get("units", 1))
        prop_type = normalize_property_type_for_backend(row.get("property_type", ""))

        subject = {
            "address": address,
            "street": address,
            "city": "",
            "state": "",
            "zip": "",
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "property_type": prop_type,
            "deal_type": "",
            "assignment_fee": 0,
            "units": units,
        }

        status = "OK"
        model_arv = None
        err = None

        # Website-style exports (top-level)
        dottid_arv = None
        dottid_arv_str = ""
        dottid_estimated_rehab = None
        dottid_estimated_rehab_str = ""
        dottid_max_offer = None
        dottid_max_offer_str = ""
        dottid_num_comps_used = 0
        comps_fields = blank_comp_fields()

        result, timeout_err, status_override = run_with_timeout(subject, args.timeout)
        if status_override:
            status = status_override
            err = timeout_err
        else:
            try:
                # Legacy scoring value
                model_arv = extract_model_arv(result)

                # Website-style extraction
                arv_obj = result.get("arv") if isinstance(result, dict) else None
                rehab_raw = result.get("rehab", {}) if isinstance(result, dict) else {}
                rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))
                dottid_estimated_rehab = rehab
                dottid_estimated_rehab_str = fmt_money(rehab)

                if not isinstance(arv_obj, dict):
                    status = "ERROR"
                    err = "Invalid ARV object (result['arv'] not a dict)."
                elif is_not_enough_comps(arv_obj):
                    status = NOT_ENOUGH_TOKEN
                    dottid_arv = NOT_ENOUGH_TOKEN
                    dottid_arv_str = NOT_ENOUGH_TOKEN
                    dottid_max_offer = None
                    dottid_max_offer_str = ""
                else:
                    arv = int(_extract_arv_value(arv_obj))
                    dottid_arv = arv
                    dottid_arv_str = fmt_money(arv)

                    mao = compute_mao_like_website(arv, rehab, subject)
                    dottid_max_offer = mao
                    dottid_max_offer_str = fmt_money(mao)

                # Comps used (post selection)
                comps_used = get_comps_used(arv_obj, result) if isinstance(arv_obj, dict) else []
                dottid_num_comps_used = len(comps_used)

                # Fill top N comps fields
                for n, c in enumerate(comps_used[:top_n], start=1):
                    if not isinstance(c, dict):
                        continue
                    sold_price = c.get("sold_price")
                    comps_fields[f"comp{n}_zpid"] = c.get("zpid")
                    comps_fields[f"comp{n}_address"] = c.get("address")
                    comps_fields[f"comp{n}_sold_price"] = sold_price
                    comps_fields[f"comp{n}_sold_price_str"] = fmt_money(sold_price) if sold_price else None
                    comps_fields[f"comp{n}_distance_miles"] = c.get("distance_miles")
                    comps_fields[f"comp{n}_beds"] = c.get("beds")
                    comps_fields[f"comp{n}_baths"] = c.get("baths")
                    comps_fields[f"comp{n}_sqft"] = c.get("sqft")
                    comps_fields[f"comp{n}_zillow_url"] = c.get("zillow_url") or c.get("url")

                # If legacy model_arv was None but website says NOT_ENOUGH, align it
                if model_arv is None and status == NOT_ENOUGH_TOKEN:
                    model_arv = NOT_ENOUGH_TOKEN

                if model_arv is None and status == "OK":
                    status = "NO_ARV_EXTRACTED"

            except Exception as e:
                status = "ERROR"
                err = str(e)

        # Decision agreement: both “not enough comps”
        if baseline_arv == NOT_ENOUGH_TOKEN or model_arv == NOT_ENOUGH_TOKEN:
            decision_total += 1
            if baseline_arv == NOT_ENOUGH_TOKEN and model_arv == NOT_ENOUGH_TOKEN:
                decision_agree += 1

        error_pct = None
        within_5_flag = None
        within_10_flag = None

        if isinstance(baseline_arv, (int, float)) and isinstance(model_arv, (int, float)) and baseline_arv > 0:
            priced_count += 1
            error_pct = (model_arv - baseline_arv) / baseline_arv * 100.0
            within_5_flag = abs(error_pct) <= 5.0
            within_10_flag = abs(error_pct) <= 10.0
            if within_5_flag:
                within_5 += 1
            if within_10_flag:
                within_10 += 1
        else:
            if model_arv == NOT_ENOUGH_TOKEN:
                no_comp_count += 1

        # Row export (keep existing + add website-matched exports)
        out_row = {
            **{k: row.get(k) for k in df.columns},
            # legacy fields
            "dottid_model_arv": model_arv,
            "status": status,
            "error": err,
            "error_pct": error_pct,
            "within_5_pct": within_5_flag,
            "within_10_pct": within_10_flag,
            # website-matched exports
            "dottid_arv": dottid_arv,
            "dottid_arv_str": dottid_arv_str,
            "dottid_estimated_rehab": dottid_estimated_rehab,
            "dottid_estimated_rehab_str": dottid_estimated_rehab_str,
            "dottid_max_offer": dottid_max_offer,
            "dottid_max_offer_str": dottid_max_offer_str,
            "dottid_num_comps_used": dottid_num_comps_used,
        }
        out_row.update(comps_fields)

        rows_out.append(out_row)

    out_df = pd.DataFrame(rows_out)
    results_path = outdir / f"results_{run_ts}.csv"
    summary_path = outdir / f"summary_{run_ts}.json"

    out_df.to_csv(results_path, index=False)

    summary = {
        "run_timestamp": run_ts,
        "total_properties": total,
        "priced_properties_count": priced_count,
        "model_not_enough_comps_count": no_comp_count,
        "within_5_pct_count": within_5,
        "within_10_pct_count": within_10,
        "within_5_pct_rate": (within_5 / priced_count) if priced_count else None,
        "within_10_pct_rate": (within_10 / priced_count) if priced_count else None,
        "decision_agreement_rate": (decision_agree / decision_total) if decision_total else None,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==============================")
    print("EVAL COMPLETE")
    print("==============================")
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print("==============================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
