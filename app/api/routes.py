from fastapi import APIRouter

from app.services.generation_service import generate_text
from app.schemas import GenerateRequest, OpenAIRequest
from app.core.model_loading import load_generator_evaluator

router = APIRouter()


@router.post("/generate", tags=["generation"])
async def generate(request: GenerateRequest):
    return generate_text(request.dict())


@router.post("/generate_openai", tags=["generation"])
async def generate_openai(request: OpenAIRequest):
    return generate_text(request.dict())


@router.get("/evaluate", tags=["evaluation"])
async def evaluate_model():
    # Placeholder for model evaluation logic
    return {"message": "This is a placeholder for the model evaluation endpoint."}


@router.get("/train", tags=["training"])
async def train_model():
    # Placeholder for model training logic
    return {"message": "This is a placeholder for the model training endpoint."}


@router.post("/check-tokens", tags=["tokenization"])
async def check_tokens(text: str):
    """
    Check the number of tokens in the given text using the specified tokenizer.

    Args:
        text (str): The text to be tokenized.
        tokenizer: The tokenizer to use for counting tokens.

    Returns:
        int: The number of tokens in the text.
    """
    tokenizer, _, _, _ = load_generator_evaluator()
    inputs = tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    return inputs['input_ids'][0].shape[0]

