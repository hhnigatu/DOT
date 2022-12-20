import os
import re
import hashlib
import itertools
import textwrap
import numpy as np

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import IFrame

from PIL import Image
from collections import Counter
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, exceptions
from pathlib import Path


def get_page_count(pdf_path: str) -> tuple[int | None, bool]:
    """Gets the page count of a PDF located at PDF_PATH, as well as a boolean
    representing whether the PDF is encrypted. Page count is returned as None
    if the PDF is encrypted and cannot be accessed."""
    with open(pdf_path, "rb") as f:
        pdf = PdfReader(f)
        if pdf.isEncrypted:
            return None, True
        else:
            return pdf.getNumPages(), False


def folder_page_count(dataset_path: str) -> tuple[int, int]:
    """Get the number of total PDF pages and encrypted files (whose page count
    cannot be determined) in DATASET_PATH."""
    total_page_count = 0
    total_encrypted_count = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    file_path = os.path.join(root, file)
                    page_count, encrypted = get_page_count(file_path)
                    if page_count is not None:
                        total_page_count += page_count
                    total_encrypted_count += int(encrypted)
                except Exception:
                    print("Could not access:", file)
    return total_page_count, total_encrypted_count


def convert_pdfs_to_images(dataset_path: str, output_path: str = "images") -> None:
    """Convert each page of each PDF in DATASET_PATH to a JPEG image, and
    write the images to OUTPUT_PATH."""
    if not os.path.exists(dataset_path + "/" + output_path):
        os.mkdir(dataset_path + "/" + output_path)

    for root, _, files in os.walk(dataset_path):
        pdfs = (file for file in files if file.endswith(".pdf"))
        for tag, file in enumerate(pdfs):
            outfile = (
                Path(dataset_path)
                / output_path
                / (
                    "__".join(root.split("/")[len(dataset_path.split("/")) :])
                    + "__"
                    + file
                )
            )
            try:
                if not outfile.exists():
                    outfile.mkdir(parents=True, exist_ok=True)
                    convert_from_path(
                        Path(root) / file,
                        fmt="jpeg",
                        output_folder=outfile,
                        output_file=file.rsplit(".", 1)[0] + str(tag),
                    )
                else:
                    print("{} already exists".format(outfile))
            except exceptions.PDFPageCountError:
                print("PDF reader could not get page counts:", root + "/" + file)
            except Exception as err:
                if err.errno == 36:
                    new_path = Path(dataset_path) / output_path / file
                    print("File name too long. Changed from", outfile, "to", new_path)
                else:
                    raise err


def hash_pages(dataset_path: str):
    global path_to_dataset  # TODO: Remove global variable
    path_to_dataset = dataset_path

    # get the path to the images from the given directiory.
    images = []

    hash_df = pd.DataFrame(columns=["page_path", "hashstring", "duplicate_pages"])
    for label_folder, _, _ in os.walk(dataset_path):
        if (
            label_folder != dataset_path
            and label_folder.split("/")[-1].split(".")[-1] == "pdf"
        ):
            for _, _, image_names in os.walk(label_folder):
                relative_image_names = []
                for image_file in image_names:
                    relative_image_names.append(label_folder + "/" + image_file)
                    file = open(label_folder + "/" + image_file, "rb")
                    image = file.read()
                    data = {
                        "page_path": label_folder + "/" + image_file,
                        "hashstring": str(hashlib.md5(image).hexdigest()),
                    }
                    new_entry = pd.DataFrame(data, index=[len(hash_df) + 1])
                    hash_df = pd.concat([hash_df, new_entry])
                images.extend(relative_image_names)
    NUM_OF_DOC = len(images)

    hash_df["file_name"] = hash_df["page_path"].apply(lambda x: x.split("/")[-2])
    hash_df.index = [i for i in range(NUM_OF_DOC)]

    return hash_df


def read_from_csv(path_to_csv, columns=["duplicate_pages"]):
    df = pd.read_csv(path_to_csv)
    for column in columns:
        df[column] = df[column].apply(
            lambda x: (
                re.sub(r"[\[\]']", "", x).split(",") if isinstance(x, str) else x
            )
        )
    return df


def HashPages(dataset_path, num_rep=2, doc_index=-2):
    convert_pdfs_to_images(dataset_path)

    hash_df = hash_pages(dataset_path)

    hash_df = find_pages_with_duplicates(hash_df, num_rep)
    hash_df.to_csv(open("./hash_df.csv", "w"))
    _ = create_replica_df(hash_df, doc_index)

    _, file_replica = collect_file_level_duplicate_info(hash_df, (dataset_path))
    collected_hash = select_representative_pages(hash_df)
    collected_hash.to_csv(open("./collected_hash.csv", "w"))

    return file_replica, collected_hash


def find_pages_with_duplicates(hash_df, num_rep=2):
    """Annotate rows in hash_df with a column "duplicate_pages" that contains
    the indices of pages that have the same hashstring as the current page.
    Only rows with at least NUM_REP duplicates are annotated."""
    assert num_rep >= 2, "It doesn't make sense to look for less than 2 duplicates."

    for idx, hash in zip(hash_df.index, hash_df["hashstring"]):
        rep = hash_df.loc[hash_df["hashstring"] == hash]
        if len(rep) >= num_rep:
            rep_index = rep.index.tolist()
            rep_index.remove(idx)
            hash_df.at[idx, "duplicate_pages"] = (
                hash_df["page_path"].iloc[rep_index].tolist()
            )
    return hash_df


def get_pages(file, replica_df, doc_index=-2):
    try_df = replica_df.loc[replica_df["file_name"] == file]
    duplicate_pages = [sub for main in try_df["duppage"].tolist() for sub in main]
    page_dict = {}
    for x in duplicate_pages:
        page = x.split("/")[-1].split(".")[doc_index].split("-")[-1].lstrip("0")
        doc = x.split("/")[doc_index]
        try:
            page_dict[doc].append(page)
        except Exception:
            page_dict[doc] = [page]
    return page_dict


def create_replica_df(hash_df, doc_index=-2):
    """
    Create a dataframe for to store the info for each page and the replica along with
    the file that the pages are from.

    Parameters:
    hash_df: the dataframe with the replica pages for each page.
    doc_index: the index for the name of the file in the path of the image.

    Returns:
    a dataframe with the replica for each page along with the file that contains the
    replica page.
    """
    replica_df = pd.DataFrame(
        columns=[
            "page",
            "file_name",
            "duppage",
            "files_with_duplicate_page",
            "dupdocPages",
        ]
    )
    replica_df["page"] = hash_df.loc[hash_df["duplicate_pages"].notnull()]["page_path"]
    replica_df["duppage"] = hash_df.loc[hash_df["duplicate_pages"].notnull()][
        "duplicate_pages"
    ]

    replica_df["file_name"] = replica_df["page"].apply(
        lambda x: x.split("/")[doc_index]
    )
    replica_df["files_with_duplicate_page"] = replica_df["duppage"].apply(
        lambda x: [y.split("/")[doc_index] for y in x]
    )

    return replica_df


def collect_file_level_duplicate_info(hash_df, dataset_path, doc_index=-2):
    """
    Save the percentage of replicated pages accross different files.

    Parameters:
    hash_df: dataframe with the pages and their duplicates.
    dataset_path: path to the pdf file to get the number of pages.
    doc_index:the index for the name of the file in the path of the image.

    Returns:
    page_replica: dataframe of each of the pages with their replicas.
    file_replica: Dataframe with the percentage of match for each of the files with
    files that contain replica pages.
    """
    dataset_path = os.path.join(dataset_path, "images")
    page_replica = create_replica_df(hash_df, doc_index)
    file_replica = pd.DataFrame(
        columns=[
            "file_name",
            "files_with_duplicate_page",
            "percentage_match",
            "matched_page_num",
        ]
    )
    file_replica["file_name"] = list(set(page_replica["file_name"]))
    page_range = {}
    for doc in file_replica["file_name"]:
        page_range[doc] = get_pages(doc, page_replica)

    for ind in file_replica.index:
        file_replica["files_with_duplicate_page"][ind] = Counter(
            [
                doc
                for docs in page_replica.loc[
                    page_replica["file_name"] == file_replica["file_name"][ind]
                ]["files_with_duplicate_page"].tolist()
                for doc in docs
            ]
        )
        file_replica["matched_page_num"][ind] = page_range[
            file_replica["file_name"][ind]
        ]
    page_num = []
    for doc in list(file_replica["file_name"]):
        page_num.append(
            len(
                [
                    name
                    for name in os.listdir(os.path.join(dataset_path, doc))
                    if os.path.isfile(os.path.join(dataset_path, doc, name))
                ]
            )
        )
    file_replica["numberOfPagesInFile"] = page_num

    for id, rep in zip(file_replica.index, file_replica["files_with_duplicate_page"]):

        file_replica.at[id, "percentage_match"] = get_percentage_sim(
            file_replica["numberOfPagesInFile"][id], rep
        )
    return page_replica, file_replica


def get_percentage_sim(page_num, dup):
    """
    Calculate the percenatge of pages that are a match accross two files.

    Parameters:
    page_num: the number of pages in the file.
    dup: the number of duplicates found in the other file.

    Returns:
    The ratio of of pages that are a match accross two files with the total number of
    pages in one file.
    """
    dup_doc = {}
    for k, v in dict(dup).items():
        if v <= page_num:
            dup_doc[k] = {"ratio": v / (page_num), "totalpages": page_num, "matched": v}
    return dup_doc


def ranges(list_page_nums):
    for _, b in itertools.groupby(
        enumerate(list_page_nums), lambda pair: pair[1] - pair[0]
    ):
        b = list(b)
        yield b[0][1], b[-1][1]


class DocumentReplicaInfo:
    def __init__(self, path, name, pagenum):
        self.path = "/home/hellina/tutorial dataset/images/" + path
        self.name = name
        self.pagenum = pagenum
        self.duplicates = []

    def create_replica(
        self,
        replica_path,
        replica_name,
        match_ratio,
        matched_pages_range,
        matched_page_num,
    ):
        class Replica:
            def __init__(self):
                self.replica_path = (
                    "/home/hellina/tutorial dataset/images/" + replica_path
                )
                self.replica_name = replica_name
                self.matched_pages_range = matched_pages_range
                self.matched_page_num = matched_page_num
                self.match_ratio = match_ratio

        return Replica()

    def add_replica(self, replica):
        if replica.replica_name not in [x.replica_name for x in self.duplicates]:
            self.duplicates.append(replica)


def threshold_by_percent(
    file_df,
    min_percentage_value=0,
    max_percentage_value=100,
    min_page_in_file=1,
    max_page_in_file=21,
):
    assert min_page_in_file > 0, (
        "File should have atleast 1 page. Please set min_page_in_file to a value "
        "greater than 0."
    )
    assert max_page_in_file <= 21, (
        "The largest number of pages in your dataset is 21. Please set value of "
        "max_page_in_file to less than or equal to 21."
    )
    assert (
        min_percentage_value >= 0 and max_percentage_value >= 0
    ), "Percentage value cannot be negative."
    assert (
        min_percentage_value <= 100 and max_percentage_value <= 100
    ), "Percentage value cannot be exceed 100%."
    percentage_cutoff = {}
    max_percentage_value = max_percentage_value / 100
    min_per = min_percentage_value / 100
    percentage_cutoff[
        (
            "Percentage Match",
            min_per * 100,
            max_percentage_value * 100,
            min_page_in_file,
            max_page_in_file,
        )
    ] = []
    rep_df = file_df.loc[file_df["numberOfPagesInFile"] >= min_page_in_file].loc[
        file_df["numberOfPagesInFile"] <= max_page_in_file
    ]
    for doc, rep, page_num, page_range in zip(
        rep_df["file_name"],
        rep_df["percentage_match"],
        rep_df["numberOfPagesInFile"],
        rep_df["matched_page_num"],
    ):
        document = DocumentReplicaInfo(
            re.sub("__", "/", doc), doc.split("__")[-1], str(page_num)
        )
        for replicas, perc in zip(list(rep.keys()), list(rep.values())):
            replicas = replicas.strip()
            if (
                min_per <= round(perc["ratio"], 2)
                and round(perc["ratio"], 2) <= max_percentage_value
            ):
                replica = document.create_replica(
                    re.sub("__", "/", replicas),
                    replicas.split("__")[-1],
                    round(perc["ratio"] * 100, 2),
                    str(
                        list(
                            sorted(
                                ranges(
                                    list(set([int(x) for x in page_range[replicas]]))
                                )
                            )
                        )
                    ),
                    file_df.loc[file_df["file_name"] == replicas][
                        "numberOfPagesInFile"
                    ].values[0],
                )
                document.add_replica(replica)
        percentage_cutoff[
            (
                "Percentage Match",
                min_per * 100,
                max_percentage_value * 100,
                min_page_in_file,
                max_page_in_file,
            )
        ].append(document)

    return percentage_cutoff


def threshold_by_number_of_matched_pages(
    file_df,
    min_page_in_file_match=1,
    max_page_in_file_match=21,
    min_page_in_file=1,
    max_page_in_file=21,
):
    assert min_page_in_file > 0, (
        "File should have atleast 1 page. Please set min_page_in_file to a value "
        "greater than 0."
    )
    assert max_page_in_file <= 21, (
        "The largest number of pages in your dataset is 21. Please set value of "
        "max_page_in_file to less than or equal to 21."
    )
    assert (
        min_page_in_file_match > 0 and max_page_in_file_match > 0
    ), "Number of pages cannot be negative or zero."
    assert min_page_in_file_match <= 21 and max_page_in_file_match <= 21, (
        "The largest number of pages in your dataset is 21. Please set value of "
        "max_page_in_file to less than or equal to 21."
    )

    page_num_cutoff = {}
    page_num_cutoff[
        (
            "Number of Matched Pages",
            min_page_in_file_match,
            max_page_in_file_match,
            min_page_in_file,
            max_page_in_file,
        )
    ] = []
    rep_df = file_df.loc[file_df["numberOfPagesInFile"] > min_page_in_file].loc[
        file_df["numberOfPagesInFile"] <= max_page_in_file
    ]

    for doc, rep, page_num, page_range in zip(
        rep_df["file_name"],
        rep_df["percentage_match"],
        rep_df["numberOfPagesInFile"],
        rep_df["matched_page_num"],
    ):
        document = DocumentReplicaInfo(
            re.sub("__", "/", doc), doc.split("__")[-1], str(page_num)
        )
        for replicas, perc in zip(list(rep.keys()), list(rep.values())):
            replicas = replicas.strip()
            if (
                int(page_num * perc["ratio"]) <= max_page_in_file_match
                and int(page_num * perc["ratio"]) >= min_page_in_file_match
            ):
                replica = document.create_replica(
                    re.sub("__", "/", replicas),
                    replicas.split("__")[-1],
                    round(perc["ratio"] * 100, 2),
                    str(
                        list(
                            sorted(
                                ranges(
                                    list(set([int(x) for x in page_range[replicas]]))
                                )
                            )
                        )
                    ),
                    file_df.loc[file_df["file_name"] == replicas][
                        "numberOfPagesInFile"
                    ].values[0],
                )
                document.add_replica(replica)

        page_num_cutoff[
            (
                "Number of Matched Pages",
                min_page_in_file_match,
                max_page_in_file_match,
                min_page_in_file,
                max_page_in_file,
            )
        ].append(document)

    return page_num_cutoff


def print_duplicate_info(list_of_replica, outputPath="./"):
    """
    Display information about the files with exact duplicates.

    Parameters:
        rep_df: dataframe with the files and their percentage of match.
    """
    for conds in list_of_replica:
        cond = list(conds.keys())[0]
        docs = list(conds.values())[0]
        for doc in docs:
            indexes = [i for i, x in enumerate(docs) if x.name == doc.name]
            if len(indexes) > 1:
                for ind in indexes[1:]:
                    for rep in docs[ind].duplicates:
                        docs[indexes[0]].add_replica(rep)
                del docs[ind]

    csv = {
        "doc_a_name": [],
        "doc_a_num_pages": [],
        "doc_a_percent_in_b": [],
        "doc_a_pages_in_b": [],
        "doc_a_path": [],
        "doc_b_name": [],
        "doc_b_num_pages": [],
        "doc_b_percent_in_a": [],
        "doc_b_pages_in_a": [],
        "doc_b_path": [],
    }
    bin_id = 0
    with open(os.path.join(outputPath, "Duplicate_Records.txt"), "w") as f:
        for conds in list_of_replica:
            cond = list(conds.keys())[0]
            docs = list(conds.values())[0]

            output = ""
            output += "*" * 45 + f"Bin {bin_id}" + "*" * 44 + "\n"
            output += f"\t\t\t\t\t\tFiles with {cond[0]} between {cond[1]}-{cond[2]}\n"
            output += (
                f"\t\t\t\t\t\tNumber of Pages in files between {cond[3]}-{cond[4]}\n"
            )

            bin_id += 1

            for doc in docs:
                for replica in doc.duplicates:
                    output += textwrap.dedent(
                        f"""\
                        {"_" * 116}
                        Doc A
                        \tName: {doc.name}
                        \tNumber of Pages: {doc.pagenum}
                        \tPercentage of pages in Doc A found in Doc B: {
                            float(replica.match_ratio)}%
                        \tNumber of pages in Doc A found in Doc B: {
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))}
                        \tPath: {doc.path}

                        Doc B
                        \tName: {replica.replica_name}
                        \tNumber of Pages: {replica.matched_page_num}
                        \tPercentage of pages in Doc B found in Doc A: {round(
                                (
                                    (float(replica.match_ratio) / 100)
                                    * float(doc.pagenum)
                                )
                                / replica.matched_page_num,
                                2,
                            )
                            * 100}%
                        \tNumber of pages in Doc B found in Doc A: {
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))}
                        \tPath: {replica.replica_path}
                        """
                    )
                    # Output the same information to a CSV so that it's accessible as a
                    # spreadsheet
                    csv["doc_a_name"].append(doc.name)
                    csv["doc_a_num_pages"].append(doc.pagenum)
                    csv["doc_a_percent_in_b"].append(float(replica.match_ratio))
                    csv["doc_a_pages_in_b"].append(
                        int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                    )
                    csv["doc_a_path"].append(doc.path)
                    csv["doc_b_name"].append(replica.replica_name)
                    csv["doc_b_num_pages"].append(replica.matched_page_num)
                    csv["doc_b_percent_in_a"].append(
                        round(
                            ((float(replica.match_ratio) / 100) * float(doc.pagenum))
                            / replica.matched_page_num,
                            2,
                        )
                        * 100
                    )
                    csv["doc_b_pages_in_a"].append(
                        int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                    )
                    csv["doc_b_path"].append(replica.replica_path)
        f.write(output)
        print(output)

        csv_df = pd.DataFrame(csv)
        csv_df.to_csv(os.path.join(outputPath, "Duplicate_Records.csv"), index=False)


def select_representative_pages(hash_df):
    """
    Get representative image for pages that are exact matches to avoid processing the
    same page multiple times.

    Parameters:
    hash_df: dataframe with all the pages and their hashstrings.

    Returns:
    collected_hash: dataframe with one page per group of replica.
    """
    collected_hash = pd.DataFrame(
        hash_df.groupby("hashstring")["page_path"].apply(lambda x: "+".join(x))
    )
    collected_hash["pages"] = collected_hash["page_path"].apply(lambda x: x.split("+"))
    collected_hash["number_of_duplicates"] = collected_hash["pages"].apply(
        lambda x: len(x)
    )
    collected_hash["representative_page"] = collected_hash["pages"].apply(
        lambda x: x[0]
    )
    return collected_hash


def color_image(source_path, match=False):
    img = cv2.imread(source_path)
    if match:
        mask = np.full_like(img, [0, 200, 0])
    else:
        mask = np.full_like(img, [200, 0, 0])
    mask = cv2.resize(mask, img.shape[1::-1])

    masked = cv2.addWeighted(img, 0.5, mask, 0.4, 0)
    return masked


def visualize_file_pairs(path_1, path_2, outputPath="./outputs/"):
    hash_df_path = "./hash_df.csv"
    if not os.path.exists(
        outputPath + path_1.split(".")[0] + "_" + path_2.split(".")[0] + "A.pdf"
    ):
        path_1 = path_1.strip()
        path_1 = re.sub("/home/hellina/tutorial dataset/images/", "", path_1)
        path_1 = re.sub("/", "__", path_1)

        path_2 = path_2.strip()
        path_2 = re.sub("/home/hellina/tutorial dataset/images/", "", path_2)
        path_2 = re.sub("/", "__", path_2)
        hash_df = read_from_csv(hash_df_path)
        path_1pages = set(hash_df.loc[hash_df["file_name"] == path_1]["page_path"])
        path_2pages = set(hash_df.loc[hash_df["file_name"] == path_2]["page_path"])
        if not path_1pages or not path_2pages:
            print(
                "The file path you provided does not have any PDF pages in it. Please "
                "check the path and try again."
            )
            return [], []

        all_pages_1 = [
            Image.fromarray(color_image(x, False)).resize((200, 300), Image.ANTIALIAS)
            for x in sorted(path_1pages)
        ]
        for pages, dups in zip(
            hash_df.loc[hash_df["file_name"] == path_1]["page_path"],
            hash_df.loc[hash_df["file_name"] == path_1]["duplicate_pages"],
        ):
            try:
                for dup in dups:
                    if dup.split("/")[-2] == path_2:
                        change = sorted(path_1pages).index(pages)
                        all_pages_1[change] = Image.fromarray(
                            color_image(pages, True)
                        ).resize((200, 300), Image.ANTIALIAS)
            except Exception:
                pass
        a_path = (
            outputPath
            + path_1.split(".")[0].split("_")[-1]
            + "_"
            + path_2.split(".")[0].split("_")[-1]
            + "A.pdf"
        )
        if len(all_pages_1) > 1:
            all_pages_1[0].save(
                a_path,
                save_all=True,
                append_images=all_pages_1[1:],
            )
        else:
            all_pages_1[0].save(a_path)

        all_pages_2 = [
            Image.fromarray(color_image(x, False)).resize((200, 300), Image.ANTIALIAS)
            for x in sorted(path_2pages)
        ]
        for pages, dups in zip(
            hash_df.loc[hash_df["file_name"] == path_2]["page_path"],
            hash_df.loc[hash_df["file_name"] == path_2]["duplicate_pages"],
        ):
            try:
                for dup in dups:
                    if dup.split("/")[-2] == path_1:
                        change = sorted(path_2pages).index(pages)
                        all_pages_2[change] = Image.fromarray(
                            color_image(pages, True)
                        ).resize((200, 300), Image.ANTIALIAS)
            except Exception:
                pass

        b_path = (
            outputPath
            + path_1.split(".")[0].split("_")[-1]
            + "_"
            + path_2.split(".")[0].split("_")[-1]
            + "B.pdf"
        )
        if len(all_pages_2) > 1:
            all_pages_2[0].save(
                b_path,
                save_all=True,
                append_images=all_pages_2[1:],
            )
        else:
            all_pages_2[0].save(b_path)

        with open("visualize.html", "w") as f:
            iframe_a = (
                f"<iframe src='{a_path}' style='width:600px;"
                " height:500px;' frameborder='0'></iframe>"
            )
            iframe_b = (
                f"<iframe src='{b_path}B.pdf' style='width:600px;"
                " height:500px;' frameborder='0'></iframe>"
            )
            f.write(
                textwrap.dedent(
                    f"""\
                <!DOCTYPE html>
                <html>
                    <body>
                        <div>
                            {iframe_a} {iframe_b}
                        </div>
                    </body>
                </html>"""
                )
            )
        f.close()

    return IFrame(a_path, width=400, height=600), IFrame(b_path, width=400, height=600)


def show_matching_pages(doc1, doc2, duplicate_df):
    """
    Plot the images of the pages that matched across two documents.

    Parameters:
    doc1: name of the first document.
    doc2: name of the second document.
    duplicate_df: dataframe with the pages and duplicate information.
    """
    for pages, dups in zip(
        duplicate_df.loc[duplicate_df["file_name"] == doc1]["page"],
        duplicate_df.loc[duplicate_df["file_name"] == doc1]["duppage"],
    ):
        for dup in dups:
            if dup.split("/")[-2] == doc2:
                img = Image.open(pages).convert("RGB")
                img2 = Image.open(dup).convert("RGB")
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.subplot(1, 2, 2)
                plt.imshow(img2)
                plt.show()


def show_unique_pages_in_file(path_1, path_2):
    """
    Plot the images of the pages that did not match across two documents.

    Parameters:
    path_1: name of the first document.
    path_2: name of the second document.
    hash_df: dataframe with all the pages.
    """
    hash_df = hash_pages(dataset_path=path_to_dataset)  # TODO: Remove global variable
    hash_df = find_pages_with_duplicates(hash_df)
    path_1pages = set(hash_df.loc[hash_df["file_name"] == path_1]["page_path"])
    commonpages = []
    for pages, dups in zip(
        hash_df.loc[hash_df["file_name"] == path_1]["page_path"],
        hash_df.loc[hash_df["file_name"] == path_1]["duplicate_pages"],
    ):
        try:
            for dup in dups:
                if dup.split("/")[-2] == path_2:
                    commonpages.append(pages)
        except Exception:
            print("no duplicates found")

    unique_pages = [page for page in path_1pages if page not in commonpages]

    print(
        f"There are {len(unique_pages)} unique pages in {path_1} that are"
        + f" not found in {path_2}"
    )

    for page in unique_pages:
        img = Image.open(page).convert("RGB")
        print(page.split("/")[-1])
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.show()
