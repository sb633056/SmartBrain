from fastapi import FastAPI
from routers import kpi, auth, health

app = FastAPI()

app.include_router(kpi.router)
app.include_router(auth.router)
app.include_router(health.router)
