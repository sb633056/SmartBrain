from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import kpi, auth, health, ai

app = FastAPI()

# ------------------------------------------
# CORS (already correct)
# ------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# CLOUDFLARE-SAFE MIDDLEWARE  (important patch)
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
app.include_router(kpi.router)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(ai.router)
