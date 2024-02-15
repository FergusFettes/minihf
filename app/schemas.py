from typing import Any
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    prompt_node: Any
    tokens_per_branch: int
    output_branches: int
