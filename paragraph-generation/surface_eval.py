import argparse
import json
import tqdm
import utils
from overlapy import OverlapyTestSet, Overlapy


def n_gram_overlap(preds, inputs, n_gram):

    overlap_ratios = []
    for input, pred in tqdm.tqdm(zip(inputs, preds), total=len(inputs)):
        testset = OverlapyTestSet("pred", min_n=n_gram, max_n=n_gram, examples=[pred.lower().split()])
        overlapy = Overlapy(testsets=[testset], dataset=[input.lower().split()], n_workers=2)
        matches = overlapy.run()
        try:
            overlap_ratios.append(len(matches) / len(set(map(tuple, list(testset.ngrams())))))
        except ZeroDivisionError:
            overlap_ratios.append(0)

    return overlap_ratios


def calculate_n_gram(evaL_seq, inputs):

    n_gram_results = {}
    for n_gram in [1, 2, 3]:
        n_gram_results[n_gram] = n_gram_overlap(evaL_seq, inputs, n_gram)

    return n_gram_results


def paragraph_count_citation_mark(evaL_seq, is_gold):

    print("Paragraph counts and citation mark checking...")

    pg_counts = []
    cm_counts = []
    if is_gold:
        splitter = '\n'
    else:
        splitter = '\n\n'
    for pred in tqdm.tqdm(evaL_seq):
        # LLama-2 and GPT 3.5 splits with '\n\n'
        pg_counts.append(len(pred.split(splitter)))
        if '[REF#1]' in pred:
            cm_counts.append(1)
        else:
            cm_counts.append(0)

    return pg_counts, cm_counts


def main(args):

    outputs, golds, prompts, concat_abstracts = utils.prepare_eval_data(args.model_output_path, args.dataset_path, args.clean_output)

    eval_dict = {'model_output': {}, 'gold': {}}
    values_dict = {'model_output': {}, 'gold': {}}

    inputs = []
    for prompt in prompts:
        # For Llama-2 prompts
        if len(prompt.split('\n<</SYS>>\n\n')) == 2:
            inputs.append(prompt.split('\n<</SYS>>\n\n')[-1].replace(' [/INST]', ''))
        # GPT type prompts
        elif len(prompt.split('\n\n')) >= 2:
            inputs.append('\n\n'.join(prompt.split('\n\n')[1:-1]))
        else:
            raise Exception('Invalid prompt.')

    for key, evaluated_sequence in zip(eval_dict, [outputs, golds]):
        n_gram_results = calculate_n_gram(evaluated_sequence, inputs)
        values_dict[key]['n-gram-overlap'] = n_gram_results
        eval_dict[key]['n-gram-overlap'] = {k: sum(n_gram_results[k]) / len(n_gram_results[k]) for k in n_gram_results}

        if key == 'gold':
            pg_counts, cm_counts = paragraph_count_citation_mark(evaluated_sequence, is_gold=True)
        else:
            pg_counts, cm_counts = paragraph_count_citation_mark(evaluated_sequence, is_gold=False)

        values_dict[key]['paragraph_count'] = pg_counts
        eval_dict[key]['paragraph_count'] = sum(pg_counts) / len(pg_counts)

        values_dict[key]['citation_mark'] = cm_counts
        eval_dict[key]['citation_mark'] = sum(cm_counts) / len(cm_counts)

        word_counts = [len(text.split()) for text in evaluated_sequence]
        values_dict[key]['word_count'] = word_counts
        eval_dict[key]['word_count'] = sum(word_counts) / len(word_counts)

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
    parser.add_argument('--clean_output', action='store_true')

    main(parser.parse_args())
