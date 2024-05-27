import pandas as pd
import argparse
from pylatexenc.latex2text import LatexNodes2Text
import tqdm
import numpy as np
from bs4 import BeautifulSoup


def rw_ignored_ref_cleaning(rw_data):

    print("Removing paragraphs that include overlooked citations...")

    idx_to_be_removed = []
    for index, row in tqdm.tqdm(rw_data.iterrows(), total=len(rw_data)):
        xml_refs = BeautifulSoup(row["paragraph_xml"]).findAll("ref", type="bibr")
        refs = row["cited_paper_marks"].split(" %%% ")
        if len(xml_refs) != len(refs):
            idx_to_be_removed.append(index)
        else:
            for xml_ref, ref in zip(xml_refs, refs):
                if str(xml_ref) != ref:
                    raise Exception("Problem in reference order")

    rw_data = rw_data.drop(idx_to_be_removed)
    print("Done.")

    return rw_data


def rw_citing_abstract_cleaning(rw_data):

    print("Removing corrupted abstracts from related work data...")

    # Cleaning bad abstracts in related work data

    with open("corrupted_abstracts.txt") as fr:
        corrupted_abstract_ids = [line[:-1] for line in fr]

    rw_data = rw_data.drop(rw_data[rw_data["acl_id"].isin(corrupted_abstract_ids)].index)

    # Some empty strings are not considered as None
    rw_data['abstract'].replace('', np.nan, inplace=True)
    rw_data['year'].replace('', np.nan, inplace=True)
    rw_data = rw_data.dropna(subset=["abstract", "year"])

    print("Done.")

    return rw_data


def examples_abstract_cleaning(example_data):

    print("Removing corrupted abstracts from example sentence data...")

    # Cleaning bad abstracts in examples data
    with open("corrupted_abstracts.txt") as fr:
        corrupted_abstract_ids = [line[:-1] for line in fr]

    example_data = example_data.drop(example_data[example_data["cited_acl_id"].isin(corrupted_abstract_ids)].index)
    example_data = example_data.drop(example_data[example_data["citing_acl_id"].isin(corrupted_abstract_ids)].index)

    # Some empty strings are not considered as None
    example_data['cited_abstract'].replace('', np.nan, inplace=True)
    example_data['cited_year'].replace('', np.nan, inplace=True)
    example_data['citing_abstract'].replace('', np.nan, inplace=True)
    example_data['citing_year'].replace('', np.nan, inplace=True)

    example_data = example_data.dropna(subset=["cited_abstract", "cited_year", "citing_abstract", "citing_year"])

    print("Done.")

    return example_data


def rw_duplicate_cleaning(rw_data, duplicate_file):

    print("Removing duplicate papers from related work data...")

    # Here papers appearing in different venues with the same title will be removed from rw dataset.
    # Those papers can be either citing or cited papers

    duplicates = pd.read_csv(duplicate_file, sep="\t", header=0)

    latex_converter = LatexNodes2Text()
    title_list = [latex_converter.latex_to_text(title).lower() for title in rw_data["title"]]

    rw_data.insert(loc=13, column="clean_title", value=title_list)

    # rw_data = rw_data.drop(rw_data[rw_data["clean_title"].isin(duplicates["clean_title"])].index)
    rw_data = rw_data.drop(rw_data[rw_data["acl_id"].isin(duplicates["acl_id"])].index)

    idx_to_be_removed = []

    for index, row in tqdm.tqdm(rw_data.iterrows(), total=len(rw_data)):
        for acl_id in row["cited_papers_acl_ids"].split(" %%% "):
            if acl_id in duplicates["acl_id"].values:
                idx_to_be_removed.append(index)
                break

    rw_data = rw_data.drop(idx_to_be_removed)

    return rw_data


def examples_duplicate_cleaning(rw_data, example_data, duplicate_file):

    print("Removing duplicate papers from example sentence data...")

    # Here papers appearing in different venues with the same title will be removed from example dataset.
    # Those papers can be either citing or cited papers

    duplicates = pd.read_csv(duplicate_file, sep="\t", header=0)

    latex_converter = LatexNodes2Text()
    citing_title_list = [latex_converter.latex_to_text(title).lower() for title in example_data["citing_title"]]
    cited_paper_set = {acl_id for papers in rw_data["cited_papers_acl_ids"] for acl_id in papers.split(" %%% ")}

    example_data["citing_clean_title"] = citing_title_list

    example_data = example_data.drop(example_data[example_data["citing_acl_id"].isin(duplicates["acl_id"])].index)
    example_data = example_data.drop(example_data[example_data["cited_acl_id"].isin(duplicates["acl_id"])].index)

    # Since we removed some cited papers in rw data for cleaning we do not need their example sentences anymore.
    example_data = example_data.drop(example_data[~example_data["cited_acl_id"].isin(cited_paper_set)].index)

    print("Done.")

    return example_data


def rw_paragraph_cleaning(rw_data):

    print("Related work data paragraph cleaning...")

    # Cleaning of problematic paragraphs
    idx_to_be_removed = []

    for index, row in tqdm.tqdm(rw_data.iterrows(), total=len(rw_data)):
        # Remove paragraphs starting without an upper letter. There are many bad paragraphs like that
        if not (row["paragraph"][0].isupper()):
            idx_to_be_removed.append(index)
        # Remove figure or table captions and to avoid possible errors
        # elif row["paragraph"][:4] == "Fig." or row["paragraph"][:6] == "Figure" or row["paragraph"][:5] == "Table":
        if "fig." in row["paragraph_xml"].lower() or "figure" in row["paragraph_xml"].lower() or "table" in row["paragraph_xml"].lower():
            idx_to_be_removed.append(index)
        # Remove paragraphs including old type non-ACL citation marks
        elif "[" in row["cited_paper_marks"] or "]" in row["cited_paper_marks"]:
            idx_to_be_removed.append(index)
        # Remove short paragraphs
        elif len(row["paragraph"].split()) < 40:
            idx_to_be_removed.append(index)

    rw_data = rw_data.drop(idx_to_be_removed)

    print("Done.")

    return rw_data


def example_sentence_cleaning(example_data):

    print("Example sentence data sentence cleaning...")

    # Cleaning of problematic paragraphs
    idx_to_be_removed = []

    for index, row in tqdm.tqdm(example_data.iterrows(), total=len(example_data)):
        # Remove sentences starting without an upper letter. There are many bad paragraphs like that
        if not (row["sentence"][0].isupper()):
            idx_to_be_removed.append(index)
        # Remove figure or table captions
        elif "fig." in row["sentence"].lower() or "figure" in row["sentence"].lower() or "table" in row["sentence"].lower():
            idx_to_be_removed.append(index)
        # Remove paragraphs including old type non-ACL citation marks
        elif "[" in row["sentence"] or "]" in row["sentence"]:
            idx_to_be_removed.append(index)
        # Remove short sentences
        elif len(row["sentence"].split()) < 10:
            idx_to_be_removed.append(index)

    example_data = example_data.drop(idx_to_be_removed)

    print("Done")

    return example_data


def add_cited_acl_ids_to_rw(rw_data, acl_metadata):

    print("Adding acl_ids of cited papers to related work data...")

    latex_converter = LatexNodes2Text()
    title_list = [latex_converter.latex_to_text(title).lower() for title in acl_metadata["title"]]
    acl_metadata["clean_title"] = title_list

    cited_paper_ids = []

    for index, row in tqdm.tqdm(rw_data.iterrows(), total=len(rw_data)):
        temp_ids = []
        corruption = False
        for title in row["cited_paper_titles"].split(" %%% "):
            temp = acl_metadata[acl_metadata["clean_title"] == title]
            if len(temp) == 1:
                temp_ids.append(temp["acl_id"].item())
            else:
                corruption = True
                break

        if not corruption:
            cited_paper_ids.append(" %%% ".join(temp_ids))
        else:
            cited_paper_ids.append(None)

    rw_data["cited_papers_acl_ids"] = cited_paper_ids
    rw_data = rw_data.dropna(subset=["cited_papers_acl_ids"])

    print("Done.")

    return rw_data


def add_cited_abstracts_to_rw(rw_data, acl_metadata):

    print("Adding abstracts of cited papers to related work data...")

    with open("corrupted_abstracts.txt") as fr:
        corrupted_abstract_ids = [line[:-1] for line in fr]

    cited_paper_abs = []

    for index, row in tqdm.tqdm(rw_data.iterrows(), total=len(rw_data)):
        temp_abs = []
        corruption = False
        for acl_id in row["cited_papers_acl_ids"].split(" %%% "):
            if acl_id in corrupted_abstract_ids:
                corruption = True
                break
            else:
                temp = acl_metadata[acl_metadata["acl_id"] == acl_id]
                if len(temp) == 1 and temp["abstract"].item() is not None and temp["abstract"].item() != "":
                    temp_abs.append(temp["abstract"].item())
                else:
                    corruption = True
                    break

        if not corruption:
            cited_paper_abs.append(" %%% ".join(temp_abs))
        else:
            cited_paper_abs.append(None)

    rw_data["cited_papers_abstracts"] = cited_paper_abs
    rw_data = rw_data.dropna(subset=["cited_papers_abstracts"])

    print("Done.")

    return rw_data


def main(args):

    rw_data = pd.read_csv(args.rw_file, sep="\t", header=0)
    example_data = pd.read_csv(args.example_file, sep="\t", header=0)
    acl_metadata = pd.read_parquet(args.acl_metadata_dir, engine='pyarrow')

    updated_rw = add_cited_acl_ids_to_rw(rw_data, acl_metadata)
    updated_rw = add_cited_abstracts_to_rw(updated_rw, acl_metadata)

    updated_rw = rw_ignored_ref_cleaning(updated_rw)
    updated_rw = rw_citing_abstract_cleaning(updated_rw)
    updated_rw = rw_duplicate_cleaning(updated_rw, args.duplicate_file)
    updated_rw = rw_paragraph_cleaning(updated_rw)

    updated_examples = examples_abstract_cleaning(example_data)
    updated_examples = examples_duplicate_cleaning(updated_rw, updated_examples, args.duplicate_file)
    updated_examples = example_sentence_cleaning(updated_examples)

    updated_rw.to_csv(args.output_path + "clean_acl_related_work_data.tsv", index=False, sep="\t")
    updated_examples.to_csv(args.output_path + "clean_acl_rw_example_sentences.tsv", index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--duplicate_file', default="title_duplicate_instances.tsv", type=str)
    parser.add_argument('--rw_file', required=True, type=str)
    parser.add_argument('--example_file', required=True, type=str)
    parser.add_argument('--acl_metadata_dir', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    main(parser.parse_args())
