from fastapi import APIRouter
from app.services.ai_engine import generate_commentary, generate_spend_guidance
import numpy as np

router = APIRouter(prefix="/ai", tags=["ai"])

def make_json_safe(obj):
    """Recursively convert numpy + non-serializable objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif obj is None:
        return None
    else:
        return obj

@router.post("/commentary")
async def commentary_route(payload: dict):
    raw_kpi = payload.get("kpi_payload", {})
    safe_kpi = make_json_safe(raw_kpi)   # << THE FIX
    result = await generate_commentary(safe_kpi)
    return result

@router.post("/spend")
async def spend_route(payload: dict):
    channel_table = payload.get("channel_table", [])
    platform_fees = payload.get("platform_fees", {})
    gross_margin = payload.get("gross_margin", None)

    result = await generate_spend_guidance(channel_table, platform_fees, gross_margin)
    return result