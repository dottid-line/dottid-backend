# orchestrator.py

import time

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
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if logger:
            logger(msg)

    t_total_start = time.perf_counter()

    log("START → run_full_underwrite()")

    # ---------------------------------------------------
    # INPUT SUMMARY
    # ---------------------------------------------------
    try:
        log(
            "INPUT → "
            f"address='{subject.get('address','')}' "
            f"beds={subject.get('beds', 0)} "
            f"baths={subject.get('baths', 0)} "
            f"sqft={subject.get('sqft', 0)} "
            f"year_built={subject.get('year_built', 0)} "
            f"property_type='{subject.get('property_type','')}' "
            f"deal_type='{subject.get('deal_type','')}' "
            f"assignment_fee='{subject.get('assignment_fee','')}' "
            f"condition='{subject.get('condition','')}'"
        )
        up_paths = subject.get("uploaded_image_paths", []) or []
        log(f"INPUT → uploaded_image_paths_count={len(up_paths)}")
    except Exception:
        pass

    # ---------------------------------------------------
    # STEP 1: ARV + COMPS
    # ---------------------------------------------------
    log("STEP 1 → Running pipeline (ARV + comps)...")

    t_step1_start = time.perf_counter()
    pipeline_out = run_pipeline(
        subject.get("address", ""),
        subject.get("beds", 0),
        subject.get("baths", 0),
        subject.get("sqft", 0),
        subject.get("year_built", 0),
        subject.get("property_type", "single_family"),
        subject,  # <-- FIX: pipeline requires the full subject too
    )
    t_step1_end = time.perf_counter()

    log("STEP 1 COMPLETE.")
    try:
        log(f"TIMING step1_pipeline_seconds={(t_step1_end - t_step1_start):.3f}")
    except Exception:
        pass

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

    try:
        if isinstance(arv_section, dict):
            arv_status = str(arv_section.get("status") or "")
            arv_msg = str(arv_section.get("message") or "")
            comps_enriched = arv_section.get("selected_comps_enriched") or []
            comps_basic = arv_section.get("selected_comps") or []
            log(
                "STEP 1 RESULT → "
                f"arv_value={arv_value} "
                f"status='{arv_status}' "
                f"message='{arv_msg}' "
                f"selected_comps_enriched_count={len(comps_enriched) if isinstance(comps_enriched, list) else 0} "
                f"selected_comps_count={len(comps_basic) if isinstance(comps_basic, list) else 0}"
            )
        scored = pipeline_out.get("scored", [])
        log(f"STEP 1 RESULT → scored_images_count={len(scored) if isinstance(scored, list) else 0}")
    except Exception:
        pass

    # ---------------------------------------------------
    # STEP 2: REHAB ESTIMATE
    # ---------------------------------------------------
    log("STEP 2 → Estimating rehab...")

    # CHANGE: Rehab must be driven by SUBJECT uploaded images (validator→room→condition),
    # not comp/Zillow-scored images from the pipeline.
    subject_image_results = []
    t_step2_classify_start = None
    t_step2_classify_end = None
    try:
        up_paths = subject.get("uploaded_image_paths", []) or []
        log(f"STEP 2 INPUT → uploaded_image_paths_count={len(up_paths)}")

        if up_paths:
            # CHANGE: lazy import to avoid crashing app on startup
            from inference_engine import classify_images
            t_step2_classify_start = time.perf_counter()
            subject_image_results = classify_images(up_paths, device="cpu", logger=log) or []
            t_step2_classify_end = time.perf_counter()
        else:
            subject_image_results = []

        valid_cnt = sum(1 for r in (subject_image_results or []) if r.get("valid") is True)
        log(
            "STEP 2 INPUT → "
            f"classify_images_total={len(subject_image_results) if isinstance(subject_image_results, list) else 0} "
            f"classify_images_valid={valid_cnt}"
        )
    except Exception as e:
        try:
            log(f"STEP 2 INPUT → classify_images_failed err='{str(e)}' (falling back to pipeline scored)")
        except Exception:
            pass
        subject_image_results = pipeline_out.get("scored", [])

    try:
        if t_step2_classify_start is not None and t_step2_classify_end is not None:
            log(f"TIMING step2_classify_images_seconds={(t_step2_classify_end - t_step2_classify_start):.3f}")
    except Exception:
        pass

    t_step2_rehab_start = time.perf_counter()
    rehab_data = estimate_rehab(
        subject,
        subject_image_results,
    )
    t_step2_rehab_end = time.perf_counter()

    log("STEP 2 COMPLETE.")
    try:
        log(f"TIMING step2_estimate_rehab_seconds={(t_step2_rehab_end - t_step2_rehab_start):.3f}")
    except Exception:
        pass

    rehab_cost = rehab_data.get("estimate_numeric", 0)

    try:
        log(
            "STEP 2 RESULT → "
            f"property_tier='{rehab_data.get('property_tier')}' "
            f"estimate_numeric={rehab_data.get('estimate_numeric')} "
            f"estimate_str='{rehab_data.get('estimate_str')}' "
            f"sqft_cost={rehab_data.get('sqft_cost')} "
            f"kitchen_cost={rehab_data.get('kitchen_cost')} "
            f"bath_cost={rehab_data.get('bath_cost')} "
            f"roof_cost={rehab_data.get('roof_cost')} "
            f"hvac_cost={rehab_data.get('hvac_cost')} "
            f"foundation_cost={rehab_data.get('foundation_cost')} "
            f"total_images={rehab_data.get('total_images')}"
        )
    except Exception:
        pass

    # ---------------------------------------------------
    # STEP 3: MAO
    # ---------------------------------------------------
    log("STEP 3 → Computing MAO...")

    t_step3_start = time.perf_counter()
    mao_str = compute_mao(
        arv_value,
        rehab_cost,
        subject.get("assignment_fee", 0),
        subject.get("deal_type", ""),
    )
    t_step3_end = time.perf_counter()

    log("STEP 3 COMPLETE.")
    try:
        log(f"TIMING step3_compute_mao_seconds={(t_step3_end - t_step3_start):.3f}")
    except Exception:
        pass

    try:
        log(
            "STEP 3 RESULT → "
            f"arv_value={arv_value} rehab_cost={rehab_cost} "
            f"assignment_fee={subject.get('assignment_fee', 0)} deal_type='{subject.get('deal_type','')}' "
            f"mao_formatted='{mao_str}'"
        )
    except Exception:
        pass

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

    try:
        log("WRAP RESULT → keys=" + ",".join(list(result.keys())))
    except Exception:
        pass

    try:
        log(f"TIMING total_run_full_underwrite_seconds={(time.perf_counter() - t_total_start):.3f}")
    except Exception:
        pass

    log("DONE → run_full_underwrite() COMPLETE.")
    return result
