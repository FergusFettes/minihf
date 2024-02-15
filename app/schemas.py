from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    tokens_per_branch: int
    output_branches: int


class OpenAIRequest(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class WeaveRequest(BaseModel):
    prompt: str
    prompt_node: bool = False
    context: str
    evaluationPrompt: str
    weave_n_tokens: int = 32
    weave_budget: int = 72
    weave_round_budget: int = 24
    weave_n_expand: int = 8
    weave_beam_width: int = 1
    weave_max_lookahead: int = 3
    weave_temperature: float = 0.25
