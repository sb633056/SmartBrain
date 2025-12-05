# Your Main App File (e.g., main.py or app.py) - MODIFIED

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# IMPORTANT: Import the new async router
from app.routers import kpi, auth, health, ai, async_jobs 

app = FastAPI(title="SmartBrain Production API")

# ------------------------------------------
# CORS (Set to production-only domain later)
# ------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ⚠️ FOUNDER NOTE: Change this to your live frontend URL before launch!
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# CLOUDFLARE-SAFE MIDDLEWARE (kept your patch)
# ------------------------------------------
@app.middleware("http")
async def add_security_headers(request, call_next):
    request.scope["headers"].append((b"user-agent", b"SmartBrainClient"))
    response = await call_next(request)
    response.headers["X-Requested-With"] = "SmartBrainFrontend"
    return response

# ------------------------------------------
# Routers
# ------------------------------------------
# Include the NEW async router first
app.include_router(async_jobs.router) 
app.include_router(kpi.router)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(ai.router)
