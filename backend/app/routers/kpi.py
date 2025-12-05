from fastapi import APIRouter
# Keep this import, as the background worker will use your build_kpi logic.
from app.services.smartbrain_engine import build_kpi 

router = APIRouter(prefix="/kpi", tags=["kpi"])

# ⚠️ REMOVED: The old @router.post("/build") endpoint is no longer here.
# All file analysis now begins with the /analyze/submit-job endpoint.

# Add any other standard GET/POST endpoints that do not involve long processing here.
