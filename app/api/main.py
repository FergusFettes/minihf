import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as api_router

from app.core.model_loading import load_generator_evaluator
from app.logging_config import setup_logging
setup_logging()


# Load the generator and evaluator. Output is cached, so subsequent calls will be fast.
_ = load_generator_evaluator()

logger = logging.getLogger(__name__)
logger.info("Starting Up!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes
app.include_router(api_router)
