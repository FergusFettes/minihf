from functools import partial
import hashlib
import time

from weave import generate_outputs
from app.core.model_loading import load_generator_evaluator, set_adapter


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
    new_tokens = int(params['tokens_per_branch'])
    n_outputs = int(params['output_branches'])
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
def weave_search(prompt, context, evaluation_prompt, weave_params):
    # TODO: Implement the weave search logic
    pass


# Placeholder for the VAE-guided generation function
def generate_guided_text(prompt, task_vector_params):
    # TODO: Implement the VAE-guided text generation logic
    pass
