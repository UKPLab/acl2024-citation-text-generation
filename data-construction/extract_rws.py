import tarfile
from bs4 import BeautifulSoup
import argparse
import tqdm
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text


def extract_related_work(acl_metadata_dir, acl_corpus_dir):

    dataset_instances = []
    latex_converter = LatexNodes2Text()
    acl_metadata = pd.read_parquet(acl_metadata_dir, engine='pyarrow')
    metadata_idx = 0
    # title_list = acl_metadata["title"].apply(lambda x: latex_converter.latex_to_text(x).lower()).to_list()
    title_list = [latex_converter.latex_to_text(title).lower() for title in acl_metadata["title"]]
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
                    bibls = refs.findAll("biblStruct")
                    divs = soup.findAll("div")
                    for div in divs:
                        if div.head is not None:
                            if div.head.getText().lower() in ["related work", "related works", "previous work", "background", "introduction and related works", "introduction and related work", "background and related work", "background and related works", "previous related work", "previous related works", "backgrounds", "previous and related work", "previous and related works"]:

                                pgs = div.findAll("p")
                                for pg in pgs:
                                    pg_refs = []
                                    pg_refs_titles = []
                                    # Find all citations in the paragraph
                                    citations = pg.findAll("ref", type="bibr")
                                    for citation in citations:
                                        # Some in-text citations has no id
                                        if "target" in citation.attrs:
                                            ref_id = citation["target"][1:]
                                            # We are checking the order of references are compatible with references ids
                                            if bibls[int(ref_id[1:])]['xml:id'] == ref_id \
                                                    and bibls[int(ref_id[1:])].title.getText().lower() in title_list:
                                                # Title of the cited paper
                                                pg_refs_titles.append(bibls[int(ref_id[1:])].title.getText().lower())
                                                pg_refs.append(str(citation))
                                            else:
                                                # To drop paragraphs with non-ACL instances or bad problematic ref id
                                                pg_refs = []
                                                pg_refs_titles = []
                                                break
                                    if len(pg_refs_titles) == len(pg_refs) and len(pg_refs) > 0:
                                        metadata = acl_metadata.iloc[metadata_idx].to_dict()
                                        del metadata["full_text"]
                                        dataset_instances.append({**metadata,
                                                                  **{"paragraph_xml": str(pg),
                                                                     "paragraph": pg.getText(),
                                                                     "cited_paper_marks": " %%% ".join(pg_refs),
                                                                     "cited_paper_titles": " %%% ".join(pg_refs_titles)
                                                                     }
                                                                  }
                                                                 )
                                # We already found related work section, other sections are not necessary
                                break

                metadata_idx += 1

    return pd.DataFrame.from_dict(dataset_instances)


def main(args):

    rw_data = extract_related_work(args.acl_metadata_dir, args.acl_corpus_dir)
    rw_data.to_csv(args.output_dir, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acl_metadata_dir', required=True, type=str)
    parser.add_argument('--acl_corpus_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)
