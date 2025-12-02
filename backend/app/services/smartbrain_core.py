# smartbrain/backend/app/services/smartbrain_core.py
from app.services.engine.kpi import build_kpi_payload

def build_kpi_payload_wrapper(df):
    # Convert if needed (ensure df is a pandas DataFrame already)
    return build_kpi_payload(df)
