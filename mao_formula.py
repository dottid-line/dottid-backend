# mao_formula.py

def compute_mao(arv, rehab_cost, assignment_fee, buyer_type):
    """
    Compute numeric + formatted MAO output.

    buyer_type:
        - "rental"    → MAO = ARV * 0.85 - rehab
        - "flip"      → MAO = ARV * 0.75 - rehab
        - "wholesale" → MAO = ARV * 0.75 - rehab - assignment_fee
    """

    def log(msg):
        try:
            print(msg, flush=True)
        except Exception:
            pass

    log("MAO → START compute_mao()")

    # Defensive parsing
    try:
        arv = float(arv) if arv is not None else 0.0
    except Exception:
        arv = 0.0

    try:
        rehab_cost = float(rehab_cost) if rehab_cost is not None else 0.0
    except Exception:
        rehab_cost = 0.0

    try:
        assignment_fee = float(assignment_fee) if assignment_fee is not None else 0.0
    except Exception:
        assignment_fee = 0.0

    buyer_type = (buyer_type or "").lower().strip()

    log(
        "MAO → INPUTS "
        f"arv={arv} rehab_cost={rehab_cost} assignment_fee={assignment_fee} buyer_type='{buyer_type}'"
    )

    # Compute numeric
    if buyer_type == "rental":
        log("MAO → Using rental formula: ARV * 0.85 - rehab")
        mao_value = arv * 0.85 - rehab_cost

    elif buyer_type == "flip":
        log("MAO → Using flip formula: ARV * 0.75 - rehab")
        mao_value = arv * 0.75 - rehab_cost

    elif buyer_type == "wholesale":
        log("MAO → Using wholesale formula: ARV * 0.75 - rehab - assignment_fee")
        mao_value = arv * 0.75 - rehab_cost - assignment_fee

    else:
        log(f"MAO → ERROR invalid buyer_type='{buyer_type}'")
        return {
            "status": "error",
            "message": f"Invalid buyer_type='{buyer_type}'",
            "mao_value": 0.0,
            "mao_formatted": "$0",
        }

    log(f"MAO → Raw computed mao_value={mao_value}")

    # prevent negative
    if mao_value < 0:
        log("MAO → Clamping negative MAO to 0")
    mao_value = max(mao_value, 0)

    # formatted string
    mao_formatted = f"${mao_value:,.0f}"

    log(f"MAO → FINAL mao_value={mao_value} mao_formatted='{mao_formatted}'")
    log("MAO → END compute_mao()")

    return {
        "status": "ok",
        "mao_value": mao_value,
        "mao_formatted": mao_formatted,
    }
