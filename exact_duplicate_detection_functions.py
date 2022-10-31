import os
import re
import torch
import hashlib
import itertools
import numpy as np
import cv2 as cv

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from IPython.display import IFrame, display


from PIL import Image
from collections import Counter
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path, exceptions


def getNumPages(path):
    """
    Paramters:
    path: the path to the pdf file

    Returns:
    sumation: sum of the number of pages of each pdf file in the folder
    enc: number of encrypted files that could not be accessed

    """
    sumation = 0
    enc = 0
    with open(path, "rb") as f:
        pdf = PdfFileReader(f)
        if pdf.isEncrypted:
            enc = enc + 1
            print(f.name)
        else:
            sumation = sumation + pdf.getNumPages()
    return sumation, enc


def NumPagesInFolders(dataset_path):
    """
    Parmater:
    dir: the directory to the folder where the pdf files are located.

    Returns:
    info: a tuple of the sum of all the pages of the pdf files in the directory and the number of pdf files that could not be accessed becuase of encryption.

    """
    info = {}
    info["numOfPages"] = 0
    info["numFilesEncrpted"] = 0
    # print('looking in ', dir)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    info["numOfPages"], info["numFilesEncrpted"] = (
                        info["numOfPages"] + getNumPages(os.path.join(root, file))[0],
                        info["numFilesEncrpted"]
                        + getNumPages(os.path.join(root, file))[1],
                    )
                except:
                    print("Could not access: ", file)
    print("Number of pages in the folder: ", info["numOfPages"])
    print("Number of encrypted files: ", info["numFilesEncrpted"])
    return info.values()


def PdfToImage(dataset_path, outputPath="images"):
    """
    Convert each page of each pdf in the given directory to jpeg format.

    Parametrs:
    dir: the directory of the folders with the PDFs
    outputPath: place to save the images

    Return:
    tagged: a list of tuples that have the identifying tag and the image object for each page.

    """
    if not os.path.exists(dataset_path + "/" + outputPath):
        os.mkdir(dataset_path + "/" + outputPath)
    # tagged=[]
    for root, dirs, files in os.walk(dataset_path):
        for tag, file in enumerate(files):
            if file.endswith(".pdf"):
                outfile = os.path.join(
                    dataset_path,
                    outputPath,
                    "__".join(root.split("/")[len(dataset_path.split("/")) :])
                    + "__"
                    + file,
                )
                try:
                    if not os.path.exists(outfile):
                        os.mkdir(outfile)
                        convert_from_path(
                            os.path.join(root, file),
                            fmt="jpeg",
                            output_folder=os.path.join(
                                dataset_path,
                                outputPath,
                                "__".join(
                                    root.split("/")[len(dataset_path.split("/")) :]
                                )
                                + "__"
                                + file,
                            ),
                            output_file=file.rsplit(".", 1)[0] + str(tag),
                        )
                    else:
                        print(outfile + " already exists")
                    # for tag, image in enumerate(images):
                #     #     tagged.append((file.rsplit( ".", 1 )[ 0 ]  + str(tag), file))
                except exceptions.PDFPageCountError:
                    print("PDF Counter could not get page numebrs: ", root + "/" + file)
                except Exception as err:
                    if err.errno == 36:
                        outfile = os.path.join(dataset_path, outputPath, file)
                        print(
                            "File name too long. Changed from:        ",
                            os.path.join(
                                dataset_path,
                                outputPath,
                                "__".join(
                                    root.split("/")[len(dataset_path.split("/")) :]
                                )
                                + "__"
                                + file,
                            ),
                        )
                        print("To:                                      ", outfile)

    #             #     os.mkdir(os.path.join(root, outputPath, file +'_'))
    #             #     images= convert_from_path(os.path.join(root, file), fmt='jpeg',  output_folder=outputPath + "/" + file , output_file=file.rsplit( ".", 1 )[ 0 ] +'_'  + str(tag))
    #             #     for tag, image in enumerate(images):
    #             #         tagged.append((file.rsplit( ".", 1 )[ 0 ]+'_'  + str(tag), file +'_'))

    # return tagged


def hash_pages(dataset_path):
    # get the path to the images from the given directiory.
    images = []
    global path_to_dataset
    path_to_dataset = dataset_path

    hash_df = pd.DataFrame(columns=["page_path", "hashstring", "duplicate_pages"])
    for label_folder, _, file_names in os.walk(dataset_path):

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
                    # data= {'page_path': label_folder+ "/" + image_file, 'whash': imagehash.whash(image), }
                    data = {
                        "page_path": label_folder + "/" + image_file,
                        "hashstring": str(hashlib.md5(image).hexdigest()),
                    }
                    new_entry = pd.DataFrame(data, index=[len(hash_df) + 1])
                    hash_df = pd.concat([hash_df, new_entry])
                images.extend(relative_image_names)
    NUM_OF_DOC = len(images)
    # hash_df['hashstring']=hash_df['md5hash'].apply(lambda x: str(x))
    hash_df["file_name"] = hash_df["page_path"].apply(lambda x: x.split("/")[-2])
    hash_df.index = [i for i in range(NUM_OF_DOC)]
    # hash_df=get_replicated(hash_df)
    hash_df.to_csv(open(path_to_dataset + "hash_df.csv", "w"))
    return hash_df
    #


def read_from_csv(path_to_csv, columns=["duplicate_pages"]):
    df = pd.read_csv(path_to_csv)
    for column in columns:
        # print(df[column].apply(lambda x: print(len(x)) if isinstance(x, str) else print(type(x), x)))
        df[column] = df[column].apply(
            lambda x: re.sub("[\[\]']", "", x).split(",") if (isinstance(x, str)) else x
        )
    return df


def HashPages(dataset_path, num_rep=2, doc_index=-2):
    # get the path to the images from the given directiory.
    images = []

    PdfToImage(dataset_path)
    hash_df = pd.DataFrame(columns=["page_path", "hashstring", "duplicate_pages"])
    for label_folder, _, file_names in os.walk(dataset_path):

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
                    # data= {'page_path': label_folder+ "/" + image_file, 'whash': imagehash.whash(image), }
                    data = {
                        "page_path": label_folder + "/" + image_file,
                        "hashstring": str(hashlib.md5(image).hexdigest()),
                    }
                    new_entry = pd.DataFrame(data, index=[len(hash_df) + 1])
                    hash_df = pd.concat([hash_df, new_entry])
                images.extend(relative_image_names)
    NUM_OF_DOC = len(images)
    # hash_df['hashstring']=hash_df['md5hash'].apply(lambda x: str(x))
    hash_df["file_name"] = hash_df["page_path"].apply(lambda x: x.split("/")[-2])
    hash_df.index = [i for i in range(NUM_OF_DOC)]
    hash_df = find_pages_with_duplicates(hash_df, num_rep)
    hash_df.to_csv(open("./hash_df.csv", "w"))
    page_replica = create_replica_df(hash_df, doc_index)
    global path_to_dataset
    path_to_dataset = dataset_path

    # hash_df=read_from_csv(dataset_path+'hash_df.csv') #to read cached hash values from CSV. Comment out and uncomment above to hash images from scratch.

    # print(hash_df)
    # page_replica=create_replica_df(hash_df, doc_index)
    page_replica, file_replica = collect_file_level_duplicate_info(
        hash_df, (dataset_path)
    )
    collected_hash = select_representative_pages(hash_df)
    collected_hash.to_csv(open("./collected_hash.csv", "w"))

    # hash_df=get_replicated(hash_df, num_rep)

    return file_replica, collected_hash


def find_pages_with_duplicates(hash_df, num_rep=2):
    """
    find the pages with the same hash value

    parameters:
    hash_df: the dataframe containing the hash values and the path of the pages.
    num_dup: minimum number of replicas that user wants to find in the dataset. By default set to 2 so that at the very least it can detect pages that are duplicates.

    """
    assert num_rep >= 2, "Minumum number of replicas allowed==2"

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
        except:
            page_dict[doc] = [page]
    return page_dict


def create_replica_df(hash_df, doc_index=-2):
    """
    create a dataframe for to store the info for each page and the replica along with the file that the
    pages are from.

    Parameters:
    hash_df: the dataframe with the replica pages for each page.
    doc_index: the index for the name of the file in the path of the image.

    Returns:
    a dataframe with the replica for each page along with the file that contains the replica page.

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
    file_replica: Dataframe with the percentage of match for each of the files with files that contain replica
    pages.

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
        # print(dataset_path+'/' +doc)
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
    The ratio of of pages that are a match accross two files with the total number of pages in one file.
    """
    dup_doc = {}
    for k, v in dict(dup).items():
        # print(v)
        # print(page_num)
        # print()
        if v <= page_num:
            dup_doc[k] = {"ratio": v / (page_num), "totalpages": page_num, "matched": v}
    # print(dup_doc)
    return dup_doc


def ranges(list_page_nums):
    for a, b in itertools.groupby(
        enumerate(list_page_nums), lambda pair: pair[1] - pair[0]
    ):
        b = list(b)
        yield b[0][1], b[-1][1]


class document_replica_info:
    def __init__(self, path, name, pagenum):
        self.path = "/home/hellina/tutorial dataset/images/" + path
        self.name = name
        self.pagenum = pagenum
        self.duplicates = []

    # def create_replica()

    def create_replica(
        self,
        replica_path,
        replica_name,
        match_ratio,
        matched_pages_range,
        matched_page_num,
    ):
        class replica:
            def __init__(self):
                self.replica_path = (
                    "/home/hellina/tutorial dataset/images/" + replica_path
                )
                self.replica_name = replica_name
                self.matched_pages_range = matched_pages_range
                self.matched_page_num = matched_page_num
                self.match_ratio = match_ratio

        return replica()

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
    assert (
        min_page_in_file > 0
    ), "File should have atleast 1 page. Please set min_page_in_file to a value greater than 0."
    assert (
        max_page_in_file <= 21
    ), "The largest number of pages in your dataset is 21. Please set value of max_page_in_file to less than or equal to 21."
    assert (
        min_percentage_value >= 0 and max_percentage_value >= 0
    ), "Percentage value cannot be negative."
    assert (
        min_percentage_value <= 100 and max_percentage_value <= 100
    ), "Percentage value cannot be exceed 100%."
    percentage_cutoff = {}
    t = min_percentage_value / 100
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
        document = document_replica_info(
            re.sub("__", "/", doc), doc.split("__")[-1], str(page_num)
        )
        # print(document.name)
        for replicas, perc in zip(list(rep.keys()), list(rep.values())):
            replicas = replicas.strip()
            if (
                min_per <= round(perc["ratio"], 2)
                and round(perc["ratio"], 2) <= max_percentage_value
            ):
                # print('rep: ', replicas.split('__')[-1])
                # print(round(perc['ratio'],2))
                # print('page_nu:', file_df.loc[file_df['file_name']==replicas.split('__')[-1]]['numberOfPagesInFile'])
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
                # print(replica.replica_name)
        percentage_cutoff[
            (
                "Percentage Match",
                min_per * 100,
                max_percentage_value * 100,
                min_page_in_file,
                max_page_in_file,
            )
        ].append(document)

    # print(percentage_cutoff[0].duplicates[0].match_ratio)
    return percentage_cutoff


def threshold_by_number_of_matched_pages(
    file_df,
    min_page_in_file_match=1,
    max_page_in_file_match=21,
    min_page_in_file=1,
    max_page_in_file=21,
):
    assert (
        min_page_in_file > 0
    ), "File should have atleast 1 page. Please set min_page_in_file to a value greater than 0."
    assert (
        max_page_in_file <= 21
    ), "The largest number of pages in your dataset is 21. Please set value of max_page_in_file to less than or equal to 21."
    assert (
        min_page_in_file_match > 0 and max_page_in_file_match > 0
    ), "Number of pages cannot be negative or zero."
    assert (
        min_page_in_file_match <= 21 and max_page_in_file_match <= 21
    ), "The largest number of pages in your dataset is 21. Please set value of max_page_in_file to less than or equal to 21."

    page_num_cutoff = {}
    t = min_page_in_file_match
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
        document = document_replica_info(
            re.sub("__", "/", doc), doc.split("__")[-1], str(page_num)
        )
        # print(document.name)
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
                # print(replica.replica_name)

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
    t: minimum threshold for the percentage of match between files.
    model: logistic regresion model to help with identifying threshold if interested in using number of matched pages as constraint.

    """
    for conds in list_of_replica:
        # =items.items()
        cond = list(conds.keys())[0]
        docs = list(conds.values())[0]
        # print(cond)
        # print(docs)
        for doc in docs:
            indexes = [i for i, x in enumerate(docs) if x.name == doc.name]
            if len(indexes) > 1:
                for ind in indexes[1:]:
                    for rep in docs[ind].duplicates:
                        docs[indexes[0]].add_replica(rep)

                del docs[ind]
    bin_id = 0
    with open(os.path.join(outputPath, "Duplicate_Records.txt"), "w") as f:
        for conds in list_of_replica:
            # =items.items()
            cond = list(conds.keys())[0]
            docs = list(conds.values())[0]
            # print(cond)
            # print(docs)
            f.write(
                "*********************************************Bin "
                + str(bin_id)
                + "********************************************\n"
            )
            print(
                "*********************************************Bin "
                + str(bin_id)
                + "********************************************\n"
            )
            bin_id += 1
            # if cond[1]==0.0:
            #     f.write("\t\t\t\t\t\t\t\t\t\tFiles with "+str(cond[0])+" Match " +cond[1] + "  " + str(cond[3]) + " and " + str(cond[4])+'\n')
            # else:
            f.write(
                "\t\t\t\t\t\tFiles with "
                + str(cond[0])
                + " between  "
                + str(cond[1])
                + "-"
                + str(cond[2])
                + "\n"
            )
            f.write(
                "\t\t\t\t\t\tNumber of Pages in files between "
                + str(cond[3])
                + "-"
                + str(cond[4])
                + "\n"
            )
            print(
                "\t\t\t\t\t\tFiles with "
                + str(cond[0])
                + " between  "
                + str(cond[1])
                + "-"
                + str(cond[2])
                + "\n"
            )
            print(
                "\t\t\t\t\t\tNumber of Pages in files between "
                + str(cond[3])
                + "-"
                + str(cond[4])
                + "\n"
            )
            # f.write("Number of Docs in Bin:  " + str(len(list_of_replica[bin_id][1]))+ "\n")
            f.write("\n")
            print("\n")
            for doc in docs:
                # print(doc)

                # for doc, rep, page_num, page_range in zip(rep_df['file_name'], rep_df['percentage_match'], rep_df['numberOfPagesInFile'], rep_df['matched_page_num']):
                # f.write("Doc Path:         " + re.sub('__','/',doc.name) + "\n")
                # f.write("Doc Name:         " + doc.path.split('__')[-1]+ "\n")
                # f.write("Number of Pages:  " + str(doc.pagenum)+ "\n")
                # f.write("\n")

                # zip(list(rep.keys()), list(rep.values())):
                for replica in doc.duplicates:
                    # print(replica.replica_name)

                    # if (min_value<perc['ratio']>=t) :
                    print(
                        "____________________________________________________________________________________________________________________\n"
                    )
                    print("Doc A\n")
                    print("\tName: " + doc.name + "\n")
                    print("\tNumber of Pages: " + str(doc.pagenum) + "\n")
                    print(
                        "\tPercentage of pages in Doc A found in Doc B: "
                        + str(float(replica.match_ratio))
                        + "%\n"
                    )
                    print(
                        "\tNumber of pages in Doc A found in Doc B: "
                        + str(
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                        )
                        + "\n"
                    )

                    print("\tPath: " + doc.path + "\n")

                    print("Doc B\n")
                    print("\tName: " + replica.replica_name + "\n")
                    print("\tNumber of Pages: " + str(replica.matched_page_num) + "\n")
                    print(
                        "\tPercentage of pages in Doc B found in Doc A: "
                        + str(
                            round(
                                (
                                    (float(replica.match_ratio) / 100)
                                    * float(doc.pagenum)
                                )
                                / replica.matched_page_num,
                                2,
                            )
                            * 100
                        )
                        + "%\n"
                    )
                    print(
                        "\tNumber of pages in Doc B found in Doc A: "
                        + str(
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                        )
                        + "\n"
                    )

                    print("\tPath: " + replica.replica_path + "\n")

                    f.write(
                        "____________________________________________________________________________________________________________________\n"
                    )
                    f.write("Doc A\n")

                    f.write("\tName: " + doc.name + "\n")
                    f.write("\tNumber of Pages: " + str(doc.pagenum) + "\n")
                    f.write(
                        "\tPercentage of pages in Doc A found in Doc B: "
                        + str(float(replica.match_ratio))
                        + "%\n"
                    )
                    f.write(
                        "\tNumber of pages in Doc A found in Doc B: "
                        + str(
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                        )
                        + "\n"
                    )
                    f.write("\tPath: " + doc.path + "\n")

                    f.write("Doc B\n")
                    f.write("\tName: " + replica.replica_name + "\n")
                    f.write(
                        "\tNumber of Pages: " + str(replica.matched_page_num) + "\n"
                    )
                    f.write(
                        "\tPercentage of pages in Doc B found in Doc A: "
                        + str(
                            round(
                                (
                                    (float(replica.match_ratio) / 100)
                                    * float(doc.pagenum)
                                )
                                / replica.matched_page_num,
                                2,
                            )
                            * 100
                        )
                        + "%\n"
                    )
                    f.write(
                        "\tNumber of pages in Doc B found in Doc A: "
                        + str(
                            int(float(replica.match_ratio) / 100 * float(doc.pagenum))
                        )
                        + "\n"
                    )
                    f.write("\tPath: " + replica.replica_path + "\n")
                    f.write("\n")
    f.close()


def select_representative_pages(hash_df):
    """
    Get representative image for pages that are exact matches to avoid processing the same page multiple times.

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


def color_image(sourcepath, match=False):
    img = cv2.imread(sourcepath)
    # mask= cv2.imread(maskpath)
    if match:
        mask = np.full_like(img, [0, 200, 0])
    else:
        mask = np.full_like(img, [200, 0, 0])
    mask = cv2.resize(mask, img.shape[1::-1])

    masked = cv2.addWeighted(img, 0.5, mask, 0.4, 0)
    # cv2.imwrite('/home/hellina/sample/masked.jpg', masked)
    return masked


def visualize_file_pairs(path_1, path_2, outputPath="./outputs/"):
    hash_df_path = "./hash_df.csv"
    # print(hash_df_path)
    if not os.path.exists(
        outputPath + path_1.split(".")[0] + "_" + path_2.split(".")[0] + "A.pdf"
    ):
        path_1 = path_1.strip()
        path_1 = re.sub("/home/hellina/tutorial dataset/images/", "", path_1)
        path_1 = re.sub("/", "__", path_1)

        path_2 = path_2.strip()
        path_2 = re.sub("/home/hellina/tutorial dataset/images/", "", path_2)
        path_2 = re.sub("/", "__", path_2)
        # hash_df=hash_pages(dataset_path=path_to_dataset)
        # hash_df=find_pages_with_duplicates(hash_df)
        hash_df = read_from_csv(hash_df_path)
        # print(hash_df['file_name'])
        # print(path_1)
        path_1pages = set(hash_df.loc[hash_df["file_name"] == path_1]["page_path"])
        path_2pages = set(hash_df.loc[hash_df["file_name"] == path_2]["page_path"])
        if not path_1pages or not path_2pages:
            print(
                "The file path you provided does not have any PDF pages in it. Please check the path and try again."
            )
            return [], []

        # print(path_1pages)
        # print(path_2pages)
        # print(path_1)
        # print()
        commonpages_1 = {}
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
                        # commonpages_2[pages]=color_image(pages, True)
            except:
                pass
        # print(len(all_pages_1))
        if len(all_pages_1) > 1:
            all_pages_1[0].save(
                outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "A.pdf",
                save_all=True,
                append_images=all_pages_1[1:],
            )
        else:
            all_pages_1[0].save(
                outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "A.pdf"
            )

        commonpages_2 = {}
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
                        # print(dup.split('/')[-2])
                        # print(path_1)
                        change = sorted(path_2pages).index(pages)
                        # print(change)
                        all_pages_2[change] = Image.fromarray(
                            color_image(pages, True)
                        ).resize((200, 300), Image.ANTIALIAS)
                        # commonpages_2[pages]=color_image(pages, True)
            except:
                pass
        # print(all_pages)

        if len(all_pages_2) > 1:
            all_pages_2[0].save(
                outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "B.pdf",
                save_all=True,
                append_images=all_pages_2[1:],
            )
        else:
            all_pages_2[0].save(
                outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "B.pdf"
            )

        with open("visualize.html", "w") as f:
            f.write(
                """
            <!DOCTYPE html>
                <html>
                <body>

                    <div> 
            
            """
            )
            f.write(
                "<iframe src='"
                + outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "A.pdf' style='width:600px; height:500px;' frameborder='0'></iframe>"
            )
            f.write(
                "<iframe src='"
                + outputPath
                + path_1.split(".")[0].split("_")[-1]
                + "_"
                + path_2.split(".")[0].split("_")[-1]
                + "B.pdf' style='width:600px; height:500px;' frameborder='0'></iframe>"
            )
            f.write(
                """
                    </div>
                </body>
                
                </html>
            """
            )
        f.close()

    filepath = (
        outputPath
        + path_1.split(".")[0].split("_")[-1]
        + "_"
        + path_2.split(".")[0].split("_")[-1]
        + "A.pdf"
    )
    filepath2 = (
        outputPath
        + path_1.split(".")[0].split("_")[-1]
        + "_"
        + path_2.split(".")[0].split("_")[-1]
        + "B.pdf"
    )
    return IFrame(filepath, width=400, height=600), IFrame(
        filepath2, width=400, height=600
    )
    # else:
    #     print("")
    # return open(path,'r').read()


def show_matching_pages(path_1, path_2, duplicate_df):
    """
    Plot the images of the pages that matched accross two documents.

    Parameters:
    doc1: name of the first document.
    doc2: name of the second document.
    duplicate_df: dataframe with the pages and duplicate information.
    """
    # print('called with: ', doc1, doc2, duplicate_df)
    for pages, dups in zip(
        duplicate_df.loc[duplicate_df["file_name"] == path_1]["page"],
        duplicate_df.loc[duplicate_df["file_name"] == path_1]["duppage"],
    ):
        for dup in dups:
            # print(dup)
            if dup.split("/")[-2] == path_2:
                img = Image.open(pages).convert("RGB")

                img2 = Image.open(dup).convert("RGB")

                plt.subplot(121), plt.imshow(img)
                plt.subplot(122), plt.imshow(img2)

                plt.show()
    # print('called with: ', doc1, doc2, duplicate_df)
    # return True


def show_unique_pages_in_file(path_1, path_2):
    """
    Plot the images of the pages that did not match accross two documents.

    Parameters:
    path_1: name of the first document.
    path_2: name of the second document.
    hash_df: dataframe with all the pages.


    """
    hash_df = hash_pages(dataset_path=path_to_dataset)
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
        except:
            print("no duplicates found")

    unique_pages = [page for page in path_1pages if page not in commonpages]
    print(
        "There are "
        + str(len(unique_pages))
        + " unique pages in "
        + path_1
        + " that are not found in "
        + path_2
    )
    for page in unique_pages:
        img = Image.open(page).convert("RGB")
        print(page.split("/")[-1])
        plt.subplot(121), plt.imshow(img)
        plt.show()
    # return False
