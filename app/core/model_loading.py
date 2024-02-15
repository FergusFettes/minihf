from functools import lru_cache
from contextlib import contextmanager
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import peft
import torch
from bigvae import DecoderOnlyTransformerVAE, VAERouter


@lru_cache
def load_generator_evaluator():
    evaluator_adapter_name = "jdpressman/minihf_evaluator_mistral_7b_v0.1"
    generator_adapter_name = None
    peft_config = peft.PeftConfig.from_pretrained(evaluator_adapter_name)
    model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(evaluator_adapter_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, evaluator_adapter_name, "evaluator")
    if generator_adapter_name:
        model.load_adapter(generator_adapter_name, "generator")
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    )
    if os.path.exists("BigVAE-Mistral-7B-v0.2"):
        vae_model = DecoderOnlyTransformerVAE(
            model, "cuda:0", peft_config, z_dim=768,
        )
        vae_model.load_pretrained("BigVAE-Mistral-7B-v0.2")
        vae_model.vae.requires_grad_(False)
        router = VAERouter(model, vae_model, "cuda:0", peft_config)
        router.load_pretrained("BigVAE-Mistral-7B-v0.2")
    else:
        vae_model = None
        router = None
    return tokenizer, model, vae_model, router


@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            print(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                print("Reached here!")
                yield model
    finally:
        model.set_adapter(old_adapter_name)
