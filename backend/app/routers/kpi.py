# app/routers/kpi.py (MODIFIED)

from fastapi import APIRouter
# Keep this import, as the worker will still use it.
# You will adapt build_kpi to run in the background.
from app.services.smartbrain_engine import build_kpi 

router = APIRouter(prefix="/kpi", tags=["kpi"])

# ⚠️ ACTION: The old @router.post("/build") endpoint is REMOVED.
# This prevents the web server from blocking on file uploads.
# All file analysis now starts via the new async_jobs router.

# If you have other KPI-related GET endpoints, keep them here.
