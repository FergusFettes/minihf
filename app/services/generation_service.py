import logging
from functools import partial
import hashlib
import time

from weave import make_score_prompt_fn, TreeNode, generate_outputs
from app.core.model_loading import load_generator_evaluator, set_adapter

logger = logging.getLogger(__name__)


def generate_text(params):
    tokenizer, model, vae_model, router = load_generator_evaluator()
    generator = (tokenizer, model)
    adapter_name = "generator" if "generator" in generator[1].peft_config else None
    generate_fn = set_adapter(generator[1], adapter_name)(partial(generate_outputs, generator, batch_size=1))

    prompt = params['prompt']
    if 'prompt_node' in params:
        prompt_node = params['prompt_node']
    else:
        prompt_node = False
    if 'tokens_per_branch' in params:
        new_tokens = int(params['tokens_per_branch'])
    elif 'max_tokens' in params:
        new_tokens = int(params['max_tokens'])
    else:
        new_tokens = 32
        logger.info("No tokens_per_branch or max_tokens provided, defaulting to 32")

    if 'output_branches' in params:
        n_outputs = int(params['output_branches'])
    elif 'n' in params:
        n_outputs = int(params['n'])
    else:
        n_outputs = 1
        logger.info("No output_branches or n provided, defaulting to 1")

    base_model_name = generator[1].active_peft_config.base_model_name_or_path
    try:
        adapter = params["adapter"]
    except KeyError:
        adapter = "generator" if "generator" in generator[1].peft_config else None
    if (adapter == "generator") or (adapter is None):
        gen_fn = generate_fn
    elif adapter == "evaluator":
        gen_fn = set_adapter(generator[1], "evaluator")(partial(generate_outputs, generator, batch_size=1))
    outs = gen_fn(prompt, new_tokens, n=n_outputs)
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
                      "base_model": base_model_name,
                      "prompt": prompt,
                      "text": out,
                      "timestamp": timestamp,
                      "nodes": []})

    return batch


# Placeholder for the weave search function
def weave_search(params):
    prompt = params['prompt']
    context = params['context']
    if 'prompt_node' in params:
        prompt_node = params['prompt_node']
    else:
        prompt_node = False
    evaluation_prompt = params['evaluationPrompt']
    full_prompt = context + " " + prompt
    tree = TreeNode(full_prompt)
    score_prompt_fn = partial(make_score_prompt_fn, evaluator)
    score_prompt_fn = partial(score_prompt_fn, evaluation_prompt)
    # MiniHF evaluator LoRA suffix
    score_prompt_fn = partial(score_prompt_fn, "<|end|>")
    # Change name to avoid overwriting global baseline evaluate_fn partial
    score_fn = partial(evaluate_fn, score_prompt_fn)
    weave_param_defaults = {"weave_n_tokens":32, "weave_budget":72,
                            "weave_round_budget":24, "weave_n_expand":8,
                            "weave_beam_width":1, "weave_max_lookahead":3,
                            "weave_temperature":0.25}
    wp = {}
    for key in weave_param_defaults.keys():
        if key in params:
            try:
                wp[key] = int(params[key])
            except ValueError:
                wp[key] = float(params[key])
        else:
            wp[key] = weave_param_defaults[key]
    branches = weave_tree_search(tree=tree,
                                 generate_fn=partial(generate_fn,
                                                     n_tokens=wp["weave_n_tokens"]),
                                 evaluate_fn=score_fn,
                                 budget=wp["weave_budget"],
                                 round_budget=wp["weave_round_budget"],
                                 n_expand=wp["weave_n_expand"],
                                 beam_width=wp["weave_beam_width"],
                                 max_lookahead=wp["weave_max_lookahead"],
                                 temperature=wp["weave_temperature"])
    batch = []
    if prompt_node:
        timestamp = str(time.time())
        id_ = hashlib.md5((prompt + timestamp).encode("UTF-8")).hexdigest()
        batch.append({"id":id_,
                      "prompt":prompt,
                      "evaluationPrompt":evaluation_prompt,
                      "text":"",
                      "timestamp":timestamp,
                      "nodes":[]})
    for branch in branches:
        branch_text = branch.branch_text()
        timestamp = str(time.time())
        id_ = hashlib.md5((branch_text + timestamp).encode("UTF-8")).hexdigest()
        batch.append({"id":id_,
                      "prompt": prompt,
                      "evaluationPrompt": evaluation_prompt,
                      "text":branch_text,
                      "timestamp":timestamp,
                      "nodes":branch.serialize_branch()})
    return batch


# Placeholder for the VAE-guided generation function
def generate_guided_text(prompt, task_vector_params):
    # TODO: Implement the VAE-guided text generation logic
    pass
