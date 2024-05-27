from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import tqdm
import json
import torch
import random
import utils

TRUE_MODEL = "google/t5_xxl_true_nli_mixture"

torch.manual_seed(0)
random.seed(0)


def main(args):

    outputs, golds, prompts, concat_abstracts = utils.prepare_eval_data(args.model_output_path, args.dataset_path, args.clean_output)

    tokenizer = AutoTokenizer.from_pretrained(TRUE_MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRUE_MODEL, torch_dtype=torch.bfloat16, device_map='auto')

    results = {'Gold -> Model Output': [], 'Abstracts -> Gold': [], 'Abstracts -> Model Output': [], }

    for gold, output in tqdm.tqdm(zip(golds, outputs), total=len(golds), desc='Gold -> Model Output'):

        input_prompt = 'premise: {} hypothesis: {}'.format(gold, output)
        tokenized_input = tokenizer(input_prompt, return_tensors='pt')

        with torch.inference_mode():
            output_sequence = model.generate(input_ids=tokenized_input['input_ids'].to('cuda'), max_new_tokens=10)

        decoded_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        result = 1 if decoded_output == '1' else 0
        results['Gold -> Model Output'].append(result)

    for concat_abstract, gold in tqdm.tqdm(zip(concat_abstracts, golds), total=len(concat_abstracts), desc='Abstracts -> Gold'):

        input_prompt = 'premise: {} hypothesis: {}'.format(concat_abstract, gold)
        tokenized_input = tokenizer(input_prompt, return_tensors='pt')

        with torch.inference_mode():
            output_sequence = model.generate(input_ids=tokenized_input['input_ids'].to('cuda'), max_new_tokens=10)

        decoded_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        result = 1 if decoded_output == '1' else 0
        results['Abstracts -> Gold'].append(result)

    for concat_abstract, output in tqdm.tqdm(zip(concat_abstracts, outputs), total=len(concat_abstracts), desc='Abstracts -> Model Output'):

        input_prompt = 'premise: {} hypothesis: {}'.format(concat_abstract, output)
        tokenized_input = tokenizer(input_prompt, return_tensors='pt')

        with torch.inference_mode():
            output_sequence = model.generate(input_ids=tokenized_input['input_ids'].to('cuda'), max_new_tokens=10)

        decoded_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        result = 1 if decoded_output == '1' else 0
        results['Abstracts -> Model Output'].append(result)

    with open(args.eval_results_path, 'w') as fw:
        json.dump({'Gold -> Model Output': sum(results['Gold -> Model Output'])/len(results['Gold -> Model Output']),
                   'Abstracts -> Gold (Baseline)': sum(results['Abstracts -> Gold'])/len(results['Abstracts -> Gold']),
                   'Abstracts -> Model Output': sum(results['Abstracts -> Model Output'])/len(results['Abstracts -> Model Output'])}, fw, indent=2)

    with open(args.eval_values_path, 'w') as fw:
        json.dump(results, fw, indent=2)


    print('Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--model_output_path', required=True, type=str)
    parser.add_argument('--eval_results_path', required=True, type=str)
    parser.add_argument('--eval_values_path', required=True, type=str)
    parser.add_argument('--clean_output', action='store_true')

    main(parser.parse_args())
