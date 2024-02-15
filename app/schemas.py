from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    tokens_per_branch: int
    output_branches: int
