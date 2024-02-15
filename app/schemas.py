from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    prompt_node: bool = False
    context: str
    new_tokens: int
    weave_beam_width: int


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
