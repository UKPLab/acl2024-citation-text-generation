import argparse
import json
import os

import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer

import configs
import utils


def generate_paragraphs(config, instances, examples, intents, categorical_intents, client):

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

    deployment_name = config.model_path

    outputs = []
    costs = []

    bar = tqdm.tqdm(range(len(prompts)), total=len(prompts))
    for i in bar:
        bar.set_description('Total cost: %f - ' % sum(costs))

        if config.prompt_type in ['type-1', 'type-3', 'type-4', 'type-5', 'type-6']:
            message = [{'role': 'system', 'content': config.system_prompt}, {'role': 'user', 'content': prompts[i]}]
            prompt_concat = config.system_prompt + '\n\n' + prompts[i]
        elif config.prompt_type in ['type-2']:
            message = [{'role': 'user', 'content': prompts[i]}]
            prompt_concat = prompts[i]
        else:
            raise Exception("Invalid prompt type.")

        response = client.chat.completions.create(model=deployment_name, messages=message, max_tokens=config.max_new_tokens, seed=config.seed)

        outputs.append({'url': masked_instances.loc[i]['url'],
                        'prompt': prompt_concat,
                        'original_paragraph': masked_instances.loc[i]['paragraph'],
                        'masked_paragraph': masked_instances.loc[i]['masked_paragraph'],
                        'output': response.choices[0].message.content})

        cost = utils.cost_calculation(prompt_price=0.003,
                                      completion_price=0.004,
                                      prompt_tokens=response.usage.prompt_tokens,
                                      completion_tokens=response.usage.completion_tokens)

        costs.append(cost)

    with open(config.output_path + 'costs.txt', 'w') as fw:
        fw.write('Total cost: ' + str(sum(costs)) + '\n')
        fw.write('Individual costs:\n')
        for cost in costs:
            fw.write(str(cost) + '\n')

    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(config.output_path + 'outputs.tsv', index=False, sep='\t')


def main(config):

    client = utils.load_gpt(config.env_file_path)

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

    generate_paragraphs(config, instances, examples, intents, categorical_intents, client)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_file_path', required=True, type=str)
    parser.add_argument('--exp_id', required=True, type=str)
    parser.add_argument('--num_examples', default=1, type=int)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--example_file', required=True, type=str)
    parser.add_argument('--intent_file', required=True, type=str)
    parser.add_argument('--categorical_intent_file', required=True, type=str)
    parser.add_argument('--model_type', default='gpt', type=str)
    parser.add_argument('--model_path', default='gpt-35-turbo-0613-16k', type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='system_prompts.json', type=str)
    parser.add_argument('--prompt_type', required=True, type=str)
    parser.add_argument('--citation_limit', default=1, type=int)
    parser.add_argument('--use_example', action='store_true')
    parser.add_argument('--use_intent', action='store_true')
    parser.add_argument('--categorical_intent', action='store_true')
    parser.add_argument('--selection_mode', default='topk')
    parser.add_argument('--max_new_tokens', default=250, type=int)
    parser.add_argument('--seed', default=0, type=int)
    # Not used
    parser.add_argument('--batch_size', default=1, type=int)

    main(configs.get_config(parser.parse_args()))
