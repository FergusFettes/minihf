#!/usr/bin/env python

"""
Fine-tunes a language model on pre-tokenized data using a Typer CLI application.
"""

import random
import json
import zipfile
from pathlib import Path
from itertools import chain, islice
import typer
import sys
from typing import Optional

import accelerate
import peft
import torch
from torch import optim
from torch.utils import data
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm, trange

app = typer.Typer()


print = tqdm.external_write_mode()(print)


# Class to handle a zipped conversation dataset
class ZippedConversationsDataset:
    """
    Dataset class to lazily load conversations from a ZIP file.
    """

    def __init__(self, zip_file: Path):
        self.training_items = []
        self.load_from_zip(zip_file)

    def load_from_zip(self, zip_file: Path):
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            if file_.endswith("/") or file_.startswith("__MACOSX"):
                continue
            with zip_.open(file_) as infile:
                training_item = self.process_file(file_, infile)
                if training_item:
                    self.training_items.append(training_item)
        random.shuffle(self.training_items)

    def process_file(self, file_name: str, file_content):
        """
        Processes a file within the ZIP file depending on the extension.
        """
        if file_name.endswith(".txt"):
            return file_content.read().decode('UTF-8')
        else:
            conversation = json.load(file_content)
            return self.extract_text_from_conversation(conversation)

    def extract_text_from_conversation(self, conversation: dict):
        """
        Extract and return text from conversation JSON dict.
        """
        for id_ in conversation["responseDict"]:
            branch = conversation["responseDict"][id_]
            if branch["rating"]:  # if True
                return branch["prompt"] + branch["text"]
        return None

    def __len__(self):
        return len(self.training_items)

    def __next__(self):
        return random.sample(self.training_items, 1)[0]


# Function to batch data into tuples of length n; Callable as a utility function
def batched(iterable, n):
    """
    Batch data into tuples of length n. The last batch may be shorter.
    Example: batched('ABCDEFG', 3) --> ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch


# Function to convert a batch of examples to tensors
def batch_to_tensors(batch, tokenizer, context, device="cpu"):
    """
    Converts a batch of text examples to input_ids and attention_mask tensors.
    """
    document_tokens = tokenizer(batch).input_ids
    for tokens in document_tokens:
        tokens.append(tokenizer.eos_token_id)
    chunks = list(batched(chain.from_iterable(document_tokens), context))
    seq_len = max(len(x) for x in chunks)
    input_ids = torch.zeros(len(chunks), seq_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(chunks), seq_len, dtype=torch.long, device=device)
    for i, x in enumerate(chunks):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        attention_mask[i, : len(x)] = 1
    return input_ids, attention_mask


# CLI main function
@app.command()
def main(
    batch_size: int = 4,
    bits: Optional[int] = typer.Option(4, help="quantization bits"),
    pretraining_dataset: str = "togethercomputer/RedPajama-Data-1T-Sample",
    user_dataset: Path = typer.Option(..., help="MiniHF user dataset path"),
    dropout: Optional[float] = None,
    epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    lr: float = 1e-4,
    model: Path = typer.Option(..., help="model name or path"),
    context: int = 2048,
    output: Path = typer.Option(..., help="path to save adapter"),
    rank: int = 8,
    start_from: Optional[str] = None
):
    """
    Fine-tunes a language model using user-provided datasets and training parameters.
    """
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=gradient_accumulation_steps
    )

    model_base, tokenizer = get_model(model, bits, accelerator)
    accelerator.wait_for_everyone()

    dropout = (0.0 if epochs == 1 else 0.1) if dropout is None else dropout
    if start_from is not None:
        model = load_adapter(start_from, dropout, accelerator, model_base)
    else:
        model = init_adapter(rank, dropout, accelerator, model_base)
    accelerator.wait_for_everyone()

    model, opt, dataloader = prepare_model_and_data(accelerator, model, tokenizer, gradient_checkpointing, lr)
    train_and_save(model, opt, dataloader, accelerator)


def get_model(model, bits, accelerator):
    """
    Handle the model loading/initialization.
    """
    if Path(model).exists():
        model_name = Path(model).resolve()
    else:
        model_name = model

    accelerator.on_main_process(f"Loading model: {model_name}", file=sys.stderr)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if accelerator.num_processes == 1 else {"": accelerator.device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model_base, tokenizer


def init_adapter(rank, dropout, accelerator, model_base):
    """
    Initialize the model adapter.
    """
    accelerator.on_main_process("Initializing adapter", file=sys.stderr)
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=8,
        lora_dropout=dropout,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "lm_head",
        ],
    )
    model = peft.get_peft_model(model_base, peft_config)
    return model


def load_adapter(start_from, dropout, accelerator, model_base):
    """
    Load an existing model adapter.
    """
    accelerator.on_main_process(f"Loading adapter: {start_from}", file=sys.stderr)
    with accelerator.main_process_first():
        model = peft.PeftModel.from_pretrained(model_base, start_from, is_trainable=True)
        model.active_peft_config.lora_dropout = dropout
        return model


def prepare_model_and_data(
        accelerator,
        model,
        tokenizer,
        gradient_checkpointing,
        lr,
        user_dataset,
        context,
        batch_size,
        pretraining_dataset
):
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if accelerator.is_main:
        model.print_trainable_parameters()

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    dataloader = get_and_prepare_data(tokenizer, user_dataset, context, batch_size, pretraining_dataset)
    return accelerator.prepare(model, opt, dataloader)


def get_and_prepare_data(tokenizer, user_dataset, context, batch_size, pretraining_dataset):
    """
    Preprocess data and set up dataloaders.
    """
    user_dataset = ZippedConversationsDataset(user_dataset)
    user_preprocessed = batch_to_tensors(user_dataset.training_items,
                                         tokenizer,
                                         context)

    pretraining_dataset = load_dataset(pretraining_dataset)
    pretraining_dataloader = data.DataLoader(
        pretraining_dataset['train']['text'],
        batch_size=1,
        shuffle=True,
    )
    user_data_tokens = user_preprocessed[0].shape[0] * context
    min_tokens = (1024 ** 2 * 5)  # Roughly ten megabytes
    pretraining_preprocessed = []
    if user_data_tokens < min_tokens:
        pt_iter = iter(pretraining_dataloader)
        pretraining_tokens = 0
        while user_data_tokens + pretraining_tokens < min_tokens:
            pretraining_texts = []
            for i in range(64):
                pretraining_texts.append(next(pt_iter))
            pretraining_texts = [i[0] for i in pretraining_texts]
            pretraining_preprocessed.append(batch_to_tensors(pretraining_texts,
                                                             tokenizer,
                                                             context))
            pretraining_tokens = sum([i[0].shape[0] * i[0].shape[1]
                                      for i in pretraining_preprocessed])

    pt_inputs = torch.cat([i[0] for i in pretraining_preprocessed])
    pt_masks = torch.cat([i[1] for i in pretraining_preprocessed])
    preprocessed = (torch.cat((user_preprocessed[0], pt_inputs)),
                    torch.cat((user_preprocessed[1], pt_masks)))
    preprocessed = zip([i for i in preprocessed[0]],
                       [i for i in preprocessed[1]])
    preprocessed = [i for i in preprocessed]
    dataloader = data.DataLoader(
        preprocessed,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def train_and_save(args, model, opt, dataloader, accelerator):
    """
    Train the model and save the weights.
    """
    i = 0
    for epoch in trange(args.epochs, disable=not accelerator.is_main_process):
        for input_ids, attention_mask in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=False,
                )
                losses = torch.nn.functional.cross_entropy(
                    outputs.logits.transpose(-1, -2),
                    input_ids[:, 1:],
                    reduction="none",
                )
                mask = attention_mask[:, :-1] * attention_mask[:, 1:]
                loss = torch.sum(losses * mask, dtype=torch.float32) / torch.sum(
                    mask, dtype=torch.float32
                )

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

                loss_global = accelerator.reduce(loss, "mean")
                accelerator.on_main_process(f"epoch: {epoch}, step: {i}, loss: {loss_global.item():g}")
                i += 1

    if accelerator.is_main_process:
        accelerator.on_main_process(f"Saving adapter to {args.output}", file=sys.stderr)
        accelerator.unwrap_model(model).save_pretrained(args.output, safe_serialization=True)


if __name__ == "__main__":
    app()
