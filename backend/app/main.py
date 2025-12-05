from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# IMPORTANT: Import the new async_jobs router
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
# CLOUDFLARE-SAFE MIDDLEWARE
# ------------------------------------------
@app.middleware("http")
async def add_security_headers(request, call_next):
    # Add user-agent so Cloudflare doesn't block the request
    request.scope["headers"].append((b"user-agent", b"SmartBrainClient"))

    response = await call_next(request)

    # Add a frontend-identifying header (Cloudflare whitelists "X-Requested-With")
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
