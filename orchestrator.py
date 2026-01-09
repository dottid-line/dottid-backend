# orchestrator.py

from pipeline import run_pipeline
from estimator import estimate_rehab
from mao_formula import compute_mao


def run_full_underwrite(subject, logger=None):
    """
    Master underwriting run:
      1) Pipeline → ARV + comps + Zillow image data
      2) Rehab → image-driven rehab calc
      3) MAO → buyer profile & assignment fee logic
    """

    def log(msg):
        if logger:
            logger(msg)

    log("START → run_full_underwrite()")

    # ---------------------------------------------------
    # STEP 1: ARV + COMPS
    # ---------------------------------------------------
    log("STEP 1 → Running pipeline (ARV + comps)...")

    pipeline_out = run_pipeline(
        subject.get("address", ""),
        subject.get("beds", 0),
        subject.get("baths", 0),
        subject.get("sqft", 0),
        subject.get("year_built", 0),
        subject.get("property_type", "single_family"),
        subject,  # <-- FIX: pipeline requires the full subject too
    )

    log("STEP 1 COMPLETE.")

    # IMPORTANT: expose the ARV OBJECT (not the whole pipeline dict) at result["arv"]
    arv_section = pipeline_out.get("arv", {})

    # robust ARV extraction
    arv_value = 0
    if isinstance(arv_section, dict):
        arv_value = arv_section.get("arv", 0)
    else:
        try:
            arv_value = float(arv_section or 0)
        except:
            arv_value = 0

    # ---------------------------------------------------
    # STEP 2: REHAB ESTIMATE
    # ---------------------------------------------------
    log("STEP 2 → Estimating rehab...")

    rehab_data = estimate_rehab(
        subject,
        pipeline_out.get("scored", []),
    )

    log("STEP 2 COMPLETE.")

    rehab_cost = rehab_data.get("estimate_numeric", 0)

    # ---------------------------------------------------
    # STEP 3: MAO
    # ---------------------------------------------------
    log("STEP 3 → Computing MAO...")

    mao_str = compute_mao(
        arv_value,
        rehab_cost,
        subject.get("assignment_fee", 0),
        subject.get("deal_type", ""),
    )

    log("STEP 3 COMPLETE.")

    # ---------------------------------------------------
    # WRAP & RETURN
    # ---------------------------------------------------
    log("WRAPPING RESULTS...")

    result = {
        "subject": subject,
        "arv": arv_section,          # <-- FIX: this is now the ARV object (has selected_comps_enriched)
        "pipeline": pipeline_out,    # <-- keep full pipeline output without changing anything else
        "rehab": rehab_data,
        "mao": {
            "mao_formatted": mao_str,
        },
    }

    log("DONE → run_full_underwrite() COMPLETE.")
    return result
