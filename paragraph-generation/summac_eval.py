import argparse
import json
import torch
import random
import utils
from summac.model_summac import SummaCConv

torch.manual_seed(0)
random.seed(0)


def main(args):

    outputs, golds, prompts, concat_abstracts = utils.prepare_eval_data(args.model_output_path, args.dataset_path, args.clean_output)

    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

    results = {}

    print('Abstracts -> Gold')
    results['Abstracts -> Gold'] = model_conv.score(concat_abstracts, golds)['scores']

    print('Gold -> Model Output')
    results['Gold -> Model Output'] = model_conv.score(golds, outputs)['scores']

    print('Abstracts -> Model Output')
    results['Abstracts -> Model Output'] = model_conv.score(concat_abstracts, outputs)['scores']

    score_input_to_gold = sum(results['Abstracts -> Gold']) / len(results['Abstracts -> Gold'])
    score_gold_to_output = sum(results['Gold -> Model Output']) / len(results['Gold -> Model Output'])
    score_input_to_output = sum(results['Abstracts -> Model Output']) / len(results['Abstracts -> Model Output'])

    with open(args.eval_results_path, 'w') as fw:
        json.dump({'SummaC: Abstracts -> Gold (Baseline)': score_input_to_gold,
                   'SummaC: Gold -> Model Output': score_gold_to_output,
                   'SummaC: Abstracts -> Model Output': score_input_to_output}, fw, indent=2)

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