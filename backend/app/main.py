from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# IMPORTANT: Import the new async_jobs router
from app.routers import kpi, auth, health, ai, async_jobs 

app = FastAPI(title="SmartBrain Production API")

# ------------------------------------------
# CORS & Middleware (as you had it)
# ------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
