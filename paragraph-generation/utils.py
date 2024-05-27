import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import util
from dotenv import load_dotenv
from openai import AzureOpenAI
import os


def load_gpt(env_file_path):

    load_dotenv(env_file_path)

    client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"), api_version="2023-12-01-preview", azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

    return client


def cost_calculation(prompt_price, completion_price, prompt_tokens, completion_tokens):

    prompt_cost = prompt_tokens * prompt_price / 1000
    completion_cost = completion_tokens * completion_price / 1000

    return prompt_cost + completion_cost


def get_file(file_path):
    return pd.read_csv(file_path, sep="\t", header=0)


def prepare_eval_data(model_output_path, dataset_path, clean_output):

    data = get_file(dataset_path)
    model_output = get_file(model_output_path)
    model_output.dropna(subset=['output'], inplace=True)
    model_output.reset_index(inplace=True)

    utilized_data = data[(data["paragraph"].isin(model_output["original_paragraph"])) & (data["url"].isin(model_output["url"]))]

    utilized_data.set_index('paragraph', inplace=True)
    utilized_data = utilized_data.reindex(index=model_output["original_paragraph"])
    utilized_data.reset_index(inplace=True)

    preds = model_output['output'].to_list()
    golds = model_output['masked_paragraph'].to_list()
    prompts = model_output['prompt'].to_list()

    if not (sum(model_output["original_paragraph"] == utilized_data["original_paragraph"]) == sum(model_output["url"] == utilized_data["url"]) == len(model_output) == len(utilized_data)):
        raise Exception('Problem between model output and utilized dataset.')

    cited_abstracts = utilized_data["cited_papers_abstracts"].to_list()

    citing_abstracts = utilized_data["abstract"].to_list()

    concat_abstracts = []

    for citing_abstract, cited_abstract in zip(citing_abstracts, cited_abstracts):
        # Considering multiple cited paper abstracts case
        concat_abstracts.append(citing_abstract + "\n" + cited_abstract.replace(" %%% ", '\n'))

    if clean_output:
        clean_preds = []
        for pred in preds:
            pred_split = pred.split('\n\n')
            if (len(pred_split) > 1) and ('sure' in pred_split[0].split()[0].lower()):
                clean_preds.append('\n\n'.join(pred_split[1:]))
            else:
                clean_preds.append('\n\n'.join(pred_split))

        outputs = clean_preds
    else:
        outputs = preds

    return outputs, golds, prompts, concat_abstracts


def data_control(instances, examples, intents, citation_limit):

    print('Checking whether given intents are valid and adjusting data according to citation limit')
    # Remove instances whose intents become NaN
    instances = instances.drop(intents[intents['intent'].isna()].index)

    citation_counts = instances['cited_papers_acl_ids'].apply(lambda x: len(x.split(' %%% ')))

    # Drop instances with citation counts more than citation limit
    instances = instances.drop(citation_counts[citation_counts > citation_limit].index)

    intents = intents[intents.index.isin(instances.index)]

    cited_paper_set = {acl_id for papers in instances["cited_papers_acl_ids"] for acl_id in papers.split(" %%% ")}

    # Since we removed some cited papers in rw data for cleaning we do not need their example sentences anymore.
    examples = examples[examples["cited_acl_id"].isin(cited_paper_set)]

    instances.reset_index(drop=True, inplace=True)
    examples.reset_index(drop=True, inplace=True)
    intents.reset_index(drop=True, inplace=True)

    return instances, examples, intents


def masking_no_example(batch):

    # We need to masking in real time for example sentences otherwise
    # we cannot know the order of citations, citation orders will be relative
    print('Masking the citation marks in target paragraph...')

    masked_paragraphs = []
    for index, row in tqdm.tqdm(batch.iterrows(), total=len(batch)):
        # Avoid generating multiple examples for the same cited paper and keep the citation order
        cited_paper_ids = list(dict.fromkeys(row['cited_papers_acl_ids'].split(' %%% ')))
        target_paragraph = row['paragraph']

        # To direct different citations marks of the same cited paper to the same masking
        mark_dict = {}
        # We try to keep the order of the citations
        for i, cited_id in enumerate(cited_paper_ids):
            mark_dict[cited_id] = '[REF#' + str(i+1) + ']'

        for cited_id, ref in zip(row['cited_papers_acl_ids'].split(' %%% '), row['cited_paper_marks'].split(' %%% ')):
            citation_mark = BeautifulSoup(ref, 'xml').getText()
            target_paragraph = target_paragraph.replace(citation_mark, mark_dict[cited_id])

        masked_paragraphs.append(target_paragraph)

    batch['masked_paragraph'] = masked_paragraphs

    return batch


def masking_w_example(batch, examples, similarity_model, num_examples, example_sampling):

    # We need to masking in real time for example sentences otherwise
    # we cannot know the order of citations, citation orders will be relative
    print('Masking the citation marks in target paragraph...')

    example_sentences = {}
    masked_paragraphs = []
    for index, row in tqdm.tqdm(batch.iterrows(), total=len(batch)):
        citing_paper_id = row['acl_id']
        # Avoid generating multiple examples for the same cited paper and keep the citation order
        cited_paper_ids = list(dict.fromkeys(row['cited_papers_acl_ids'].split(' %%% ')))
        target_paragraph = row['paragraph']

        example_sentences[index] = {}
        # To direct different citations marks of the same cited paper to the same masking
        mark_dict = {}
        # We try to keep the order of the citations
        for i, cited_id in enumerate(cited_paper_ids):
            mark_dict[cited_id] = '[REF#' + str(i+1) + ']'
            example_sentences[index][i+1] = []

            candidates = examples[(examples['cited_acl_id'] == cited_id) & (examples['citing_acl_id'] != citing_paper_id)]

            if len(candidates) == 0:
                continue
            else:
                # Sample examples according to given number of example parameter
                if example_sampling == 'random':
                    selected_examples = examples[(examples['cited_acl_id'] == cited_id) & (examples['citing_acl_id'] != citing_paper_id)].sample(n=num_examples, random_state=42)
                elif example_sampling == 'topk':
                    target_paragraph_emb = similarity_model.encode(target_paragraph, convert_to_tensor=True)
                    example_emb = similarity_model.encode(list(examples[(examples['cited_acl_id'] == cited_id) & (examples['citing_acl_id'] != citing_paper_id)]['sentence']), convert_to_tensor=True)

                    cosine_scores = util.cos_sim(target_paragraph_emb, example_emb)[0]

                    topk_indexes = sorted(range(len(cosine_scores)), key=lambda k: cosine_scores[k], reverse=True)

                    selected_examples = examples[(examples['cited_acl_id'] == cited_id) & (examples['citing_acl_id'] != citing_paper_id)].iloc[topk_indexes[0:num_examples]]

                else:
                    raise Exception('Problem in example sampling method.')

                # Masking actual citation mark as [REF#] and others as [OTH]
                for index_e, row_e in selected_examples.iterrows():
                    sentence = row_e['sentence']
                    target_citation_mark = BeautifulSoup(row_e['citation_mark'], 'xml').getText()
                    all_refs = BeautifulSoup(row_e['paragraph_xml'], 'xml').findAll('ref', type='bibr')

                    sentence = sentence.replace(target_citation_mark, mark_dict[cited_id])

                    for ref in all_refs:
                        if ref.getText() != target_citation_mark and ref.getText() != '':
                            sentence = sentence.replace(ref.getText(), '[OTH]')

                    # Distinguish chained [OTH] references from [REF#]
                    words = sentence.split()
                    for idx, word in enumerate(words):
                        if '[OTH]' in word and '[REF#' in word:
                            words[idx] = word.replace('[OTH]', '')

                        elif '[OTH][OTH]' in word:
                            words[idx] = '[OTH]'
                    example_sentences[index][i+1].append(' '.join(words))
                    # For control purposes: example_sentences[index][i + 1].append(sentence)

        for cited_id, ref in zip(row['cited_papers_acl_ids'].split(' %%% '), row['cited_paper_marks'].split(' %%% ')):
            citation_mark = BeautifulSoup(ref, 'xml').getText()
            target_paragraph = target_paragraph.replace(citation_mark, mark_dict[cited_id])

        masked_paragraphs.append(target_paragraph)

    batch['masked_paragraph'] = masked_paragraphs

    return batch, example_sentences


def prepare_prompt_type_1(masked_batch, masked_examples, intent_batch, categorical_intent_batch, system_prompt, prompt_plan, model_type):

    print('Preparing prompts ...')

    b_inst, e_inst = "[INST] ", " [/INST]"
    b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
    prompt_batch = []
    for index, row in tqdm.tqdm(masked_batch.iterrows(), total=len(masked_batch)):
        # Avoid repeating the same title and abstract for the same paper
        acl_ids = list(dict.fromkeys(row['cited_papers_acl_ids'].split(' %%% ')))
        abstracts = list(dict.fromkeys(row['cited_papers_abstracts'].split(' %%% ')))

        if not (len(acl_ids) == len(abstracts)):
            raise Exception('Problem in the number of cited papers in a paragraph:', index)

        if model_type == 'llama-2':
            prompt = b_inst + b_sys + system_prompt + e_sys
            prompt += 'Main paper abstract: ' + row['abstract']
        elif model_type == 'gpt':
            prompt = 'Main paper abstract: ' + row['abstract']
        else:
            raise Exception('Wrong model type')

        for i in range(len(abstracts)):
            prompt += '\n\nRelevant paper abstract: ' + abstracts[i]

        if prompt_plan['use_intent']:
            if prompt_plan['categorical_intent']:
                prompt += '\n\nIntent: ' + categorical_intent_batch.loc[index]['intent']
            else:
                prompt += '\n\nIntent: ' + intent_batch.loc[index]['intent']

        if prompt_plan['use_example']:
            for relative_id in masked_examples[index]:
                for example in masked_examples[index][relative_id]:

                    prompt += '\n\nExample: ' + example

        if model_type == 'llama-2':
            prompt += e_inst
        elif model_type == 'gpt':
            prompt += "\n\nYour related work paragraph: "
        else:
            raise Exception('Wrong model type')

        prompt_batch.append(prompt)

    return prompt_batch


def prepare_prompt_type_2(masked_batch, masked_examples, intent_batch, categorical_intent_batch, system_prompt, prompt_plan, model_type):

    print('Preparing prompts ...')
    b_inst, e_inst = "[INST] ", " [/INST]"
    b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
    instructions = system_prompt.split('%%%')
    prompt_batch = []
    for index, row in tqdm.tqdm(masked_batch.iterrows(), total=len(masked_batch)):
        # Avoid repeating the same title and abstract for the same paper
        acl_ids = list(dict.fromkeys(row['cited_papers_acl_ids'].split(' %%% ')))
        abstracts = list(dict.fromkeys(row['cited_papers_abstracts'].split(' %%% ')))

        if not (len(acl_ids) == len(abstracts)):
            raise Exception('Problem in the number of cited papers in a paragraph:', index)

        if model_type == 'llama-2-chat':
            prompt = b_inst + b_sys + e_sys
            prompt += instructions[0]
        elif model_type == 'gpt':
            prompt = instructions[0]
        else:
            raise Exception('Wrong model type')

        prompt += '\n\n' + row['abstract']
        prompt += '\n\n' + instructions[1]
        for i in range(len(abstracts)):
            prompt += '\n\n' + abstracts[i]

        if (prompt_plan['use_example']) and (prompt_plan['use_intent']):
            prompt += '\n\n' + instructions[2]
            if prompt_plan['categorical_intent']:
                prompt += '\n\n' + categorical_intent_batch.loc[index]['intent']
            else:
                prompt += '\n\n' + intent_batch.loc[index]['intent']
            prompt += '\n\n' + instructions[3]
            for relative_id in masked_examples[index]:
                for example in masked_examples[index][relative_id]:
                    prompt += '\n\n' + example

        elif (prompt_plan['use_example']) and (not prompt_plan['use_intent']):
            prompt += '\n\n' + instructions[2]
            for relative_id in masked_examples[index]:
                for example in masked_examples[index][relative_id]:
                    prompt += '\n\n' + example
        elif (not prompt_plan['use_example']) and (prompt_plan['use_intent']):
            prompt += '\n\n' + instructions[2]
            if prompt_plan['categorical_intent']:
                prompt += '\n\n' + categorical_intent_batch.loc[index]['intent']
            else:
                prompt += '\n\n' + intent_batch.loc[index]['intent']

        prompt += '\n\n' + instructions[-1]

        if model_type == 'llama-2-chat':
            prompt += e_inst
        elif model_type == 'gpt':
            prompt += "\n\nYour related work paragraph: "
        else:
            raise Exception('Wrong model type')

        prompt_batch.append(prompt)

    return prompt_batch
