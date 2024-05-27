from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, logging
import torch


def llama2(model_path):

    logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False, padding_side="left")

    # 8bit
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=True)

    # No 8bit
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
