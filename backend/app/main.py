from fastapi import FastAPI
from app.routers import kpi, auth, health, ai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kpi.router)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(ai.router)

