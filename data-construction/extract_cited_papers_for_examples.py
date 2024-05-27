import tarfile
from bs4 import BeautifulSoup
import argparse
import tqdm
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text
import spacy


def scispacy_extract_citation_sentences(data):

    print("Sentence segmentation...")

    nlp = spacy.load("en_core_sci_sm")

    citation_sentences = []

    data_iter = tqdm.tqdm(data.iterrows(), desc="Extracting citation sentences from paragraphs")

    for i, row in data_iter:
        found = False
        citation_mark = BeautifulSoup(row["citation_mark"], "xml").getText()
        for sent_no, sent in enumerate(list(nlp(row["paragraph"]).sents)):
            if citation_mark in sent.text:
                citation_sentences.append(sent.text)
                found = True
                break

        if not found:
            citation_sentences.append(None)

    data.insert(loc=1, column="sentence", value=citation_sentences)

    print("Done.")

    return data.dropna(subset=["sentence"])


def get_cited_paper_info(acl_metadata_dir, rw_file_dir):

    print("Obtaining metadata of the cited papers...")

    latex_converter = LatexNodes2Text()

    acl_metadata = pd.read_parquet(acl_metadata_dir, engine='pyarrow')
    raw_rw_data = pd.read_csv(rw_file_dir, sep="\t", header=0, dtype={"number": "string"})
    cited_paper_titles = {title for papers in raw_rw_data["cited_paper_titles"] for title in papers.split(" %%% ")}

    title_list = [latex_converter.latex_to_text(title).lower() for title in acl_metadata["title"]]
    acl_metadata["clean_title"] = title_list

    cited_paper_metadata = acl_metadata[acl_metadata["clean_title"].isin(cited_paper_titles)]

    # Some papers are seen in different venues with the same titles, to avoid ambiguity, they will not be included in examples.
    # Duplicate papers will be also removed from rw dataset.
    duplicates = cited_paper_metadata[cited_paper_metadata["clean_title"].duplicated(keep=False)]

    # duplicates.to_csv("duplicate_cited_paper_instances.tsv", index=False, sep="\t")

    cited_paper_metadata = cited_paper_metadata.drop(duplicates.index)

    print("Done.")

    return cited_paper_metadata


def extract_examples(acl_metadata_dir, acl_corpus_dir, cited_paper_metadata):

    print("Extracting examples from the raw corpus...")

    acl_metadata = pd.read_parquet(acl_metadata_dir, engine='pyarrow')

    rw_section_titles = ["related work", "related works", "previous work", "background", "introduction and related works",
                         "introduction and related work", "background and related work", "background and related works",
                         "previous related work", "previous related works", "backgrounds", "previous and related work",
                         "previous and related works"]

    example_instances = []
    example_counts = {}
    metadata_idx = 0
    with tarfile.open(acl_corpus_dir) as fr:
        files = fr.getmembers()
        # First item is only folder name, no xml file
        files = tqdm.tqdm(files[1:])
        for file in files:
            files.set_description("Processing %s file" % file.name[17:-8])
            # Order of the files and metadata are the same
            # But there are instances that exist in files but not in metadata
            # Keep the order with metadata_idx
            if file.name[17:-8] == acl_metadata["acl_id"].iloc[metadata_idx]:
                f = fr.extractfile(file)
                soup = BeautifulSoup(f, "xml")
                refs = soup.find("div", type="references")
                if refs is not None:
                    # We first get local citation ids of the papers that are cited by current examined paper
                    cited_bib_ids = {"#"+bib["xml:id"]: bib.title.getText().lower() for bib in refs.findAll("biblStruct") if bib.title.getText().lower() in cited_paper_metadata["clean_title"].values}
                    divs = soup.findAll("div")
                    for div in divs:
                        if div.head is not None:
                            # We ara looking for rw section examples.
                            if div.head.getText().lower() in rw_section_titles:
                                pgs = div.findAll("p")
                                for pg in pgs:
                                    # Find all citations in the paragraph
                                    citations = pg.findAll("ref", type="bibr")
                                    for citation in citations:
                                        # Some in-text citations has no id
                                        if "target" in citation.attrs:
                                            if citation["target"] in cited_bib_ids:
                                                # We find the corresponding cited paper metadata
                                                cited_metadata = cited_paper_metadata[cited_paper_metadata["clean_title"] == cited_bib_ids[citation["target"]]].to_dict("r")[0]
                                                citing_metadata = acl_metadata.iloc[metadata_idx].to_dict()
                                                del cited_metadata["full_text"]
                                                del citing_metadata["full_text"]

                                                # Change the column names in order not to confuse
                                                for key in list(cited_metadata.keys()):
                                                    cited_metadata["cited_"+key] = cited_metadata.pop(key)

                                                for key in list(citing_metadata.keys()):
                                                    citing_metadata["citing_"+key] = citing_metadata.pop(key)

                                                # For example id indexing
                                                try:
                                                    example_counts[cited_metadata["cited_acl_id"]] += 1
                                                except:
                                                    example_counts[cited_metadata["cited_acl_id"]] = 1

                                                example_instances.append({**{"example_id": cited_metadata["cited_acl_id"] + "%" +str(example_counts[cited_metadata["cited_acl_id"]]),
                                                                             "paragraph_xml": str(pg),
                                                                             "paragraph": pg.getText(),
                                                                             "citation_mark": str(citation)
                                                                             },
                                                                          **cited_metadata,
                                                                          **citing_metadata
                                                                          }
                                                                         )
                metadata_idx += 1

    print("Done.")

    return pd.DataFrame.from_dict(example_instances)


def main(args):

    cited_paper_metadata = get_cited_paper_info(args.acl_metadata_dir, args.rw_file_dir)
    example_pgs = extract_examples(args.acl_metadata_dir, args.acl_corpus_dir, cited_paper_metadata)
    examples = scispacy_extract_citation_sentences(example_pgs)
    examples.to_csv(args.output, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acl_metadata_dir', required=True, type=str)
    parser.add_argument('--acl_corpus_dir', required=True, type=str)
    parser.add_argument('--rw_file_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)