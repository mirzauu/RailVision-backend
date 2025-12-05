from fastapi import APIRouter

router = APIRouter()

@router.get("/stats")
def get_stats():
    return {}
