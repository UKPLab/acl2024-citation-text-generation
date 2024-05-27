import requests
import pandas as pd
import argparse
import ml_collections
import tqdm
import os
import json
import time
from transformers import AutoTokenizer

API_TOKEN = ''


def get_instances(input_file):
    return pd.read_csv(input_file, sep="\t", header=0)


def query(api_url, payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


def generate_intents(config, instances):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    with open(config.output_path + 'config.json', 'w') as fw:
        json.dump(config.to_dict(), fw)

    tokenizer = AutoTokenizer.from_pretrained('allenai/multicite-multilabel-scibert')

    for i in tqdm.tqdm(range(0, len(instances), config.batch_size)):

        inputs = list(instances['paragraph'].iloc[i:i + config.batch_size])

        # To avoid exceeding max sequence length error
        tokenized_inputs = tokenizer(inputs, truncation=True, max_length=512)
        decoded_inputs = tokenizer.batch_decode(tokenized_inputs['input_ids'], skip_special_tokens=True)

        response = query(api_url=config.model_url, payload={'inputs': decoded_inputs, 'options': {'wait_for_model': True}})
        while not isinstance(response, list):
            time.sleep(3)
            response = query(api_url=config.model_url, payload={'inputs': decoded_inputs, 'options': {'wait_for_model': True}})

        output = [output[0]['label'] for output in response]

        data = pd.DataFrame({'url': instances['url'].iloc[i:i + config.batch_size],
                             'instance': instances['paragraph'].iloc[i:i + config.batch_size],
                             'intent': output})

        data.to_csv(config.output_path + 'categorical_intentions.tsv',
                    mode='a',
                    index=False,
                    sep='\t',
                    header=not os.path.exists(config.output_path + 'categorical_intentions.tsv'))

        time.sleep(1)


def main(config):

    instances = get_instances(config.input_file)
    generate_intents(config, instances)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_url', default='https://api-inference.huggingface.co/models/allenai/multicite-multilabel-scibert', type=str)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    conf = ml_collections.ConfigDict()

    conf.model_url: str = args.model_url
    conf.input_file: str = args.input_file
    conf.output_path: str = args.output_path
    conf.batch_size: str = args.batch_size

    main(conf)
