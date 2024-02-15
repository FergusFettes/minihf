from pydantic import BaseModel

# Define a Pydantic model for the generation request
class GenerateRequest(BaseModel):
    prompt: str
    tokens_per_branch: int
    output_branches: int
    # Add other fields as needed

# You can define other request and response models here as needed.