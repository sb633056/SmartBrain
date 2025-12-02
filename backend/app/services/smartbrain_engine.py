# smartbrain/backend/app/services/smartbrain_engine.py
import pandas as pd
from app.services.smartbrain_core import build_kpi_payload_wrapper

async def build_kpi(file):
    # file is fastapi UploadFile
    df = pd.read_excel(file.file)   # keep your existing line
    result = build_kpi_payload_wrapper(df)
    return result
