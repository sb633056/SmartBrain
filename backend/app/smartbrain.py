def build_kpi_payload(df):
    # For now, return something simple
    return {
        "message": "SmartBrain backend is working!",
        "rows_received": len(df)
    }
