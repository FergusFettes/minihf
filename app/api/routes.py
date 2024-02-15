from fastapi import APIRouter

from app.services.generation_service import generate_text
from app.schemas import GenerateRequest

router = APIRouter()


@router.get("/generate", tags=["generation"])
async def generate(request: GenerateRequest):
    return generate_text(request.dict())


@router.get("/evaluate", tags=["evaluation"])
async def evaluate_model():
    # Placeholder for model evaluation logic
    return {"message": "This is a placeholder for the model evaluation endpoint."}


@router.get("/train", tags=["training"])
async def train_model():
    # Placeholder for model training logic
    return {"message": "This is a placeholder for the model training endpoint."}
