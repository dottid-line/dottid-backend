# mao_formulas.py

def compute_mao(arv, rehab_cost, assignment_fee, buyer_type):
    """
    Compute numeric + formatted MAO output.

    buyer_type:
        - "rental"    → MAO = ARV * 0.85 - rehab
        - "flip"      → MAO = ARV * 0.75 - rehab
        - "wholesale" → MAO = ARV * 0.75 - rehab - assignment_fee
    """

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

    # Compute numeric
    if buyer_type == "rental":
        mao_value = arv * 0.85 - rehab_cost

    elif buyer_type == "flip":
        mao_value = arv * 0.75 - rehab_cost

    elif buyer_type == "wholesale":
        mao_value = arv * 0.75 - rehab_cost - assignment_fee

    else:
        return {
            "status": "error",
            "message": f"Invalid buyer_type='{buyer_type}'",
            "mao_value": 0.0,
            "mao_formatted": "$0",
        }

    # prevent negative
    mao_value = max(mao_value, 0)

    # formatted string
    mao_formatted = f"${mao_value:,.0f}"

    return {
        "status": "ok",
        "mao_value": mao_value,
        "mao_formatted": mao_formatted,
    }
