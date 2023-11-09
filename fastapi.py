import time
import hashlib
from contextlib import contextmanager
from functools import partial

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from lora_tune import lora_tune_evaluator
from dataset import ZippedConversationsDataset


class GenerateParam(BaseModel):
    prompt: str
    prompt_node: bool = False
    context: str
    new_tokens: int
    weave_beam_width: int


class WeaveParam(BaseModel):
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


class CheckTokensParam(BaseModel):
    text: str


@contextmanager
def set_adapter(model, adapter_name):
    """
    A context manager that sets the specified adapter on a model for the duration of the context.
    If no adapter_name is provided, it will disable adapters.
    Restores the previous adapter setting after the context ends.
    :param model: The model on which to set an adapter.
    :param adapter_name: The name of the adapter to use.
    """
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)


def load_generator_evaluator():
    """
    Initializes and loads a pre-trained language model with adapters
        for generating text and evaluating text.
    The model is configured to use a specific evaluation adapter and
        can optionally load a generator adapter.
    Bit-and-Bytes configuration for quantization and efficient computation
        is applied to the model.
    Returns a tokenizer and the model with the necessary configurations
        for text generation and evaluation tasks.
    """
    evaluator_adapter_name = "jdpressman/minihf_evaluator_mistral_7b_v0.1"
    generator_adapter_name = ""

    peft_config = peft.PeftConfig.from_pretrained(evaluator_adapter_name)
    model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(evaluator_adapter_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, evaluator_adapter_name, "evaluator")
    if generator_adapter_name:
        model.load_adapter(generator_adapter_name, "generator")
    return tokenizer, model


def load_models():
    global evaluator, evaluate_fn, generator, generate_fn
    evaluator = generator = load_generator_evaluator()

    adapter_name = "generator" if "generator" in generator[1].peft_config else None
    generate_fn = set_adapter(generator[1], adapter_name)(partial(generate_outputs, generator, batch_size=1))
    evaluate_fn = set_adapter(evaluator[1], "evaluator")(partial(evaluate_outputs, evaluator))


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
def generate(params: GenerateParam):
    prompt = params.prompt
    prompt_node = params.prompt_node
    context = params.context
    full_prompt = context + " " + prompt
    new_tokens = params.new_tokens
    n_outputs = params.weave_beam_width
    outs = generate_fn(full_prompt, new_tokens, n=n_outputs)
    batch = []
    if prompt_node:
        timestamp = str(time.time())
        id_ = hashlib.md5((prompt + timestamp).encode("UTF-8")).hexdigest()
        batch.append({"id": id_,
                      "prompt": prompt,
                      "text": "",
                      "timestamp": timestamp,
                      "nodes": []})
    for out in outs:
        timestamp = str(time.time())
        id_ = hashlib.md5(out.encode("UTF-8")).hexdigest()
        batch.append({"id": id_,
                      "prompt": prompt,
                      "text": out,
                      "timestamp": timestamp,
                      "nodes": []})
    return JSONResponse(content=batch)


@app.post("/weave")
def weave(params: WeaveParam):
    prompt = params.prompt
    context = params.context
    prompt_node = params.prompt_node
    evaluation_prompt = params.evaluation_prompt
    full_prompt = context + " " + prompt
    tree = TreeNode(full_prompt)
    score_prompt_fn = partial(make_score_prompt_fn, evaluator)
    score_prompt_fn = partial(score_prompt_fn, evaluation_prompt)
    # MiniHF evaluator LoRA suffix
    score_prompt_fn = partial(score_prompt_fn, "<|end|>")
    # Change name to avoid overwriting global baseline evaluate_fn partial
    score_fn = partial(evaluate_fn, score_prompt_fn)

    branches = weave_tree_search(tree=tree,
                                 generate_fn=partial(generate_fn,
                                                     n_tokens=params.weave_n_tokens),
                                 evaluate_fn=score_fn,
                                 budget=params.weave_budget,
                                 round_budget=params.weave_round_budget,
                                 n_expand=params.weave_n_expand,
                                 beam_width=params.weave_beam_width,
                                 max_lookahead=params.weave_max_lookahead,
                                 temperature=params.weave_temperature)
    batch = []
    if prompt_node:
        timestamp = str(time.time())
        id_ = hashlib.md5((prompt + timestamp).encode("UTF-8")).hexdigest()
        batch.append({"id": id_,
                      "prompt": prompt,
                      "evaluationPrompt": evaluation_prompt,
                      "text": "",
                      "timestamp": timestamp,
                      "nodes": []})
    for branch in branches:
        branch_text = branch.branch_text()
        timestamp = str(time.time())
        id_ = hashlib.md5((branch_text + timestamp).encode("UTF-8")).hexdigest()
        batch.append({"id": id_,
                      "prompt": prompt,
                      "evaluationPrompt": evaluation_prompt,
                      "text": branch_text,
                      "timestamp": timestamp,
                      "nodes": branch.serialize_branch()})
    return JSONResponse(content=batch)


@app.post("/check-tokens")
def check_tokens(params: CheckTokensParam):
    text = params.text
    tokenizer, model = generator
    inputs = tokenizer([text] * 1, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    return JSONResponse(content={"total_tokens": inputs['input_ids'][0].shape[0]})


@app.post("/train-reward-model")
async def train_reward_model(file_: UploadFile = File(...)):
    unload_models()

    data = ZippedConversationsDataset(await file_.read())
    lora_tune_evaluator(data)
    torch.cuda.empty_cache()

    load_models()

    return JSONResponse(content={"status": "training complete"})


def unload_models():
    global generator
    del generator
    global evaluator
    del evaluator
    global generate_fn
    del generate_fn
    global evaluate_fn
    del evaluate_fn
    torch.cuda.empty_cache()
