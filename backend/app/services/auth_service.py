import pandas as pd
from app.smartbrain import build_kpi_payload

async def build_kpi(file):
    df = pd.read_excel(file.file)
    return build_kpi_payload(df)
