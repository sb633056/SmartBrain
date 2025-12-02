from fastapi import APIRouter, UploadFile
from app.services.smartbrain_engine import build_kpi

router = APIRouter(prefix="/kpi", tags=["kpi"])

@router.post("/build")
async def build_kpi_route(file: UploadFile):
    return await build_kpi(file)
