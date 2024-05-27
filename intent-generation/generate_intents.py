import json
import os
import argparse

import pandas as pd
import tqdm
import torch

import configs
import model_init


def get_instances(input_file):
    return pd.read_csv(input_file, sep="\t", header=0)


def get_examples(example_file):
    return pd.read_csv(example_file, sep="\t", header=0)


def generate_intents(config, instances, examples, model, tokenizer):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    with open(config.output_path + 'config.json', 'w') as fw:
        json.dump(config.to_dict(), fw)

    strategy_dict = {
        "greedy": {"do_sample": False, "num_beams": 1, "early_stopping": False, "top_k": 0, "temperature": 1,
                   "penalty_alpha": 0},
        "sampling": {"do_sample": True, "num_beams": 1, "early_stopping": False, "top_k": config.top_k,
                     "temperature": config.temperature, "penalty_alpha": 0},
        "beam": {"do_sample": False, "num_beams": config.num_beams, "early_stopping": True, "top_k": 0,
                 "temperature": 1, "penalty_alpha": 0},
        "beam_sample": {"do_sample": True, "num_beams": config.num_beams, "early_stopping": True,
                        "top_k": config.top_k, "temperature": config.temperature, "penalty_alpha": 0},
        "contrastive": {"do_sample": False, "num_beams": 1, "early_stopping": False, "top_k": config.top_k,
                        "temperature": 1, "penalty_alpha": config.penalty_alpha}}

    example_prompt = ""
    if config.num_examples > 0:
        if config.custom_examples:
            selected_examples = examples.iloc[[0, 3, 1]]
        else:
            selected_examples = examples.sample(config.num_examples)
        for index, row in selected_examples.iterrows():
            example_prompt += config.task_prefix + row['paragraph'] + config.task_suffix + row['intent'] + '\n'

    for i in tqdm.tqdm(range(0, len(instances), config.batch_size)):
        inputs = tokenizer([example_prompt + config.task_prefix + pg + config.task_suffix for pg in instances["paragraph"].iloc[i:i + config.batch_size]],
                           return_tensors="pt",
                           padding='longest')

        output_sequences = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                          attention_mask=inputs["attention_mask"].to("cuda"),
                                          max_new_tokens=config.max_new_tokens,
                                          do_sample=strategy_dict[config.strategy]["do_sample"],
                                          num_beams=strategy_dict[config.strategy]["num_beams"],
                                          early_stopping=strategy_dict[config.strategy]["early_stopping"],
                                          top_k=strategy_dict[config.strategy]["top_k"],
                                          temperature=strategy_dict[config.strategy]["temperature"],
                                          penalty_alpha=strategy_dict[config.strategy]["penalty_alpha"])

        outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        data = pd.DataFrame({"url": instances["url"].iloc[i:i + config.batch_size],
                             "prompt": example_prompt,
                             "instance": instances["paragraph"].iloc[i:i + config.batch_size],
                             "intent": outputs})

        data.to_csv(config.output_path + "intentions.tsv",
                    mode="a",
                    index=False,
                    sep="\t",
                    header=not os.path.exists(config.output_path + "intentions.tsv"))

    torch.cuda.empty_cache()


def main(config):
    if config.model == 'google/flan-t5-xxl':
        model, tokenizer = model_init.load_model(config.model)
    else:
        raise Exception('Invalid model name.')

    instances = get_instances(config.input_file)
    examples = get_examples(config.example_file)
    generate_intents(config, instances, examples, model, tokenizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', required=True, type=str)
    parser.add_argument('--model', default='google/flan-t5-xxl', type=str)
    parser.add_argument('--num_examples', default=0, type=int)
    parser.add_argument('--custom_examples', default=False, type=bool)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--example_file', default='example_intents.tsv', type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--strategy', default='greedy', type=str)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--penalty_alpha', default=0, type=float)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--max_input_len', deafult=1024, type=int)
    parser.add_argument('--max_new_tokens', default=50, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    config = configs.get_config(parser.parse_args())
    main(config)
