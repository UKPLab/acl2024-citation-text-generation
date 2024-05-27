import argparse
import evaluate
from bleurt import score
import bert_score
import json
import utils


def main(args):

    outputs, golds, prompts, concat_abstracts = utils.prepare_eval_data(args.model_output_path, args.dataset_path, args.clean_output)

    eval_dict = {'abstract_baseline': {}, 'model_performance': {}}
    values_dict = {'abstract_baseline': {}, 'model_performance': {}}

    for metric in ['rouge', 'bertscore', 'scibertscore', 'bleurt']:
        if metric == 'rouge':
            model = evaluate.load('rouge')
        elif metric == 'bertscore':
            # HF implementation is in efficient for BERTScore. It loads model for each call of compute function
            model = bert_score.BERTScorer(lang='en') # Roberta-large is default model. Best model is microsoft/deberta-xlarge-mnli
        elif metric == 'scibertscore':
            model = bert_score.BERTScorer(lang='en-sci')
        elif metric == 'bleurt':
            model = score.BleurtScorer(args.bleurt_checkpoint)

        for key, evaluated_sequence in zip(eval_dict, [concat_abstracts, outputs]):
            if metric == 'rouge':
                print('ROUGE evaluation: {}'.format(key))
                rouge_results = model.compute(predictions=evaluated_sequence, references=golds, use_aggregator=False)

                values_dict[key][metric+'-1'] = rouge_results[metric+'1']
                values_dict[key][metric+'-2'] = rouge_results[metric+'2']
                values_dict[key][metric+'-l'] = rouge_results[metric+'L']

                eval_dict[key][metric+'-1'] = sum(rouge_results[metric+'1']) / len(rouge_results[metric+'1'])
                eval_dict[key][metric+'-2'] = sum(rouge_results[metric+'2']) / len(rouge_results[metric+'2'])
                eval_dict[key][metric+'-l'] = sum(rouge_results[metric+'L']) / len(rouge_results[metric+'L'])

            elif metric == 'bertscore':
                print('BERTScore evaluation: {}'.format(key))
                bertscore_results = model.score(evaluated_sequence, golds, verbose=True)

                values_dict[key][metric] = bertscore_results[-1].tolist()
                eval_dict[key][metric] = sum(bertscore_results[-1].tolist()) / len(bertscore_results[-1])

            elif metric == 'scibertscore':
                print('SciBERTScore evaluation: {}'.format(key))
                scibertscore_results = model.score(evaluated_sequence, golds, verbose=True)

                values_dict[key][metric] = scibertscore_results[-1].tolist()
                eval_dict[key][metric] = sum(scibertscore_results[-1].tolist()) / len(scibertscore_results[-1])

            elif metric == 'bleurt':
                print('BLEURT evaluation: {}'.format(key))
                bleurt_results = model.score(references=golds, candidates=evaluated_sequence, batch_size=64)

                values_dict[key]['bleurt'] = bleurt_results
                eval_dict[key]['bleurt'] = sum(bleurt_results) / len(bleurt_results)

    with open(args.eval_results_path, 'w') as fw:
        json.dump(eval_dict, fw, indent=2)

    with open(args.eval_values_path, 'w') as fw:
        json.dump(values_dict, fw, indent=2)

    print('Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--model_output_path', required=True, type=str)
    parser.add_argument('--eval_results_path', required=True, type=str)
    parser.add_argument('--eval_values_path', required=True, type=str)
    parser.add_argument("--bleurt_checkpoint", required=True, type=str)
    parser.add_argument('--clean_output', action='store_true')

    main(parser.parse_args())
