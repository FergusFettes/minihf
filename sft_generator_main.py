```python
def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4, help="microbatch size")
    parser.add_argument(
        "--bits", type=int, choices=[4, 8, 16], default=4, help="quantization bits"
    )
    parser.add_argument("--pretraining-dataset",
                        default="togethercomputer/RedPajama-Data-1T-Sample",
                        help="bulk pretraining dataset to tune on")
    parser.add_argument("--user-dataset", type=Path, help="MiniHF user dataset path")
    parser.add_argument("--dropout", type=float, help="dropout rate")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=True, help="use gradient checkpointing"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model", type=str, required=True, help="model name")
    parser.add_argument("--context", type=int, default=2048, help="context window length")
    parser.add_argument("--output", type=Path, required=True, help="path to save adapter")
    parser.add_argument("--rank", type=int, default=8, help="the lora rank")
    parser.add_argument("--start-from", type=str, help="start from existing lora")
    return parser.parse_args()

def get_model(args, accelerator):
    """
    Handle the model loading/initialization.
    """
    if Path(args.model).exists():
        model_name = Path(args.model).resolve()
    else:
        model_name = args.model

    print0(f"Loading model: {model_name}", file=sys.stderr)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if accelerator.num_processes == 1 else {"": device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return model_base, tokenizer

def init_adapter(args, dropout, accelerator, model_base):
    """
    Initialize the model adapter.
    """
    print0("Initializing adapter", file=sys.stderr)
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.rank,
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

def load_adapter(args, dropout, accelerator, model_base):
    """
    Load an existing model adapter.
    """
    print0(f"Loading adapter: {args.start_from}", file=sys.stderr)
    with accelerator.main_process_first():
        model = peft.PeftModel.from_pretrained(model_base, args.start_from, is_trainable=True)
        if args.dropout is not None:
            model.active_peft_config.lora_dropout = dropout
        return model

def get_and_prepare_data(args, user_dataset, pretraining_dataset):
    """
    Preprocess data and set up dataloaders.
    """
    pretraining_dataloader = data.DataLoader(
        pretraining_dataset['train']['text'],
        batch_size=1,
        shuffle=True,
    )
    user_data_tokens = user_preprocessed[0].shape[0] * args.context
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
                                                             args.context))
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
        batch_size=args.batch_size,
        shuffle=True,
    )
    return dataloader

def train_and_save(args, model, opt, dataloader, accelerator):
    """
    Train the model and save the weights.
    """
    i = 0
    for epoch in trange(args.epochs, disable=not is_main):
        for input_ids, attention_mask in tqdm(dataloader, disable=not is_main):
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
                print0(f"epoch: {epoch}, step: {i}, loss: {loss_global.item():g}")
                i += 1

    if is_main:
        print0(f"Saving adapter to {args.output}", file=sys.stderr)
        accelerator.unwrap_model(model).save_pretrained(args.output, safe_serialization=True)

def main():
    args = parse_args()
    accelerator = setup_accelerator(args)
    print0 = accelerator.on_main_process(print)
    model_base, tokenizer = get_model(args, accelerator)
    dropout = calculate_dropout(args)
    if args.start_from is not None:
        model = load_adapter(args, dropout, accelerator, model_base)
    else:
        model = init_adapter(args, dropout, accelerator, model_base)
    model, opt, dataloader = prepare_model_and_data(args, model, accelerator)
    train_and_save(args, model, opt, dataloader, accelerator)
```
