import json
import os
import argparse
import random

import pandas as pd
import tqdm
import torch

import model_init
import utils
import configs
from sentence_transformers import SentenceTransformer

def generate_paragraphs(config, instances, examples, intents, categorical_intents, model, tokenizer):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    with open(config.output_path + 'config.json', 'w') as fw:
        json.dump(config.to_dict(), fw, indent=2)

    if config.prompt_plan['use_example']:
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        masked_instances, masked_examples = utils.masking_w_example(instances.copy(), examples, similarity_model, config.num_examples, config.selection_mode)
    else:
        masked_instances = utils.masking_no_example(instances.copy())
        masked_examples = {}

    if config.prompt_type in ['type-1', 'type-3', 'type-4', 'type-5', 'type-6']:
        # Structure of these prompts are the same, all use the system prompt
        prompts = utils.prepare_prompt_type_1(masked_instances, masked_examples, intents, categorical_intents, config.system_prompt, config.prompt_plan, config.model_type)
    elif config.prompt_type in ['type-2']:
        prompts = utils.prepare_prompt_type_2(masked_instances, masked_examples, intents, categorical_intents, config.system_prompt, config.prompt_plan, config.model_type)
    else:
        raise Exception("Invalid prompt type.")

    outputs = []
    for i in tqdm.tqdm(range(0, len(prompts), config.batch_size), total=len(prompts)//config.batch_size+1):

        selected_prompts = {'valid_id': [], 'prompts': []}
        if i+config.batch_size > len(prompts):
            end_index = len(prompts)
        else:
            end_index = i+config.batch_size

        for idx in range(i, end_index):
            if len(tokenizer(prompts[idx], return_tensors='pt')['input_ids'][0]) <= 1024:
                selected_prompts['prompts'].append(prompts[idx])
                selected_prompts['valid_id'].append(idx)

        inputs = tokenizer(selected_prompts['prompts'], return_tensors='pt', padding='max_length', max_length=1024)

        with torch.inference_mode():
            output_sequences = model.generate(input_ids=inputs['input_ids'].to('cuda'), max_new_tokens=config.max_new_tokens)

        decoded_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        for valid_id, output in zip(selected_prompts['valid_id'], decoded_outputs):

            outputs.append({'url': masked_instances.loc[valid_id]['url'],
                            'prompt': prompts[valid_id],
                            'original_paragraph': masked_instances.loc[valid_id]['paragraph'],
                            'masked_paragraph': masked_instances.loc[valid_id]['masked_paragraph'],
                            'output': output.replace(prompts[valid_id], '')})

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(config.output_path + 'outputs.tsv', index=False, sep='\t')


def main(config):

    torch.manual_seed(0)
    random.seed(0)

    model, tokenizer = model_init.llama2(config.model_path)

    if config.prompt_plan['use_example']:
        if config.num_examples < 1:
            raise Exception('use_example flag is True but num_example is less than 1.')
        if config.selection_mode not in ['topk', 'random']:
            raise Exception('Example selection method should be topk or random.')

    instances = utils.get_file(config.input_file)
    examples = utils.get_file(config.example_file)
    intents = utils.get_file(config.intent_file)
    categorical_intents = utils.get_file(config.categorical_intent_file)
    instances, examples, intents = utils.data_control(instances, examples, intents, config.citation_limit)

    generate_paragraphs(config, instances, examples, categorical_intents, intents, model.eval(), tokenizer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Not used
    parser.add_argument('--env_file_path', default='', type=str)
    parser.add_argument('--exp_id', required=True, type=str)
    parser.add_argument('--num_examples', default=1, type=int)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--example_file', required=True, type=str)
    parser.add_argument('--intent_file', required=True, type=str)
    parser.add_argument('--categorical_intent_file', required=True, type=str)
    parser.add_argument('--model_type', default='llama-2', type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='system_prompts.json', type=str)
    parser.add_argument('--prompt_type', required=True, type=str)
    parser.add_argument('--citation_limit', default=1, type=int)
    parser.add_argument('--use_example', action='store_true')
    parser.add_argument('--use_intent', action='store_true')
    parser.add_argument('--categorical_intent', action='store_true')
    parser.add_argument('--selection_mode', default='topk')
    parser.add_argument('--max_new_tokens', defeault=250, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    main(configs.get_config(parser.parse_args()))
