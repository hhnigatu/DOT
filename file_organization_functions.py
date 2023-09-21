import numpy as np
import os
import shutil
import re
import tqdm
import string

import pandas as pd


def set_entity_breakdown(df, column, entities):
    """
    Set the values of a column if the entities in the list appear within the text of
    each page. In this function, we break down the entities into separate
    "words" based on space.

    Parameters:
    df: the dataframe that has all the information of the pages.
    column: the column of the data frame where we would like to set values.
    entities: the list of entities that have been detected from the form or from NER.

    """
    collect_val = []
    for ind in tqdm.tqdm(df.index):
        entityintext = set()
        for entity in entities:
            for i, val in enumerate(
                x
                in re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", entity.lower()
                ).split()
                for x in re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", df["Text"][ind].lower()
                ).split()
            ):
                if val:
                    entityintext.add(
                        re.sub(
                            "[%s]" % re.escape(string.punctuation), " ", df["Text"][ind]
                        ).split()[i]
                    )
        for entity in entities:
            if all(
                x in list(entityintext)
                for x in re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", entity
                ).split()
            ):
                collect_val.append(entity)
            else:
                collect_val.append(None)
    df.insert(2, column, collect_val, allow_duplicates=False)
    return df


def set_entity_whole(df, column, entities):
    collect_val = []
    for ind in tqdm.tqdm(df.index):
        entityintext = set()
        for entity in entities:
            if re.search(
                re.sub("[%s]" % re.escape(string.punctuation), " ", entity),
                re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", df["Text"][ind].lower()
                ),
            ) or re.search(
                re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", "".join(entity.split())
                ),
                re.sub(
                    "[%s]" % re.escape(string.punctuation), " ", df["Text"][ind].lower()
                ),
            ):
                entityintext.add(entity)
        if len(entityintext) > 0:
            collect_val.append(list(entityintext))
        else:
            collect_val.append(None)
    df.insert(2, column, collect_val, allow_duplicates=False)
    return df


def set_entities_doc(page_df, doc_df, column):
    """Set the values of the names, case numbers and dates found in each document from
    the values of each page.

    Parameters:
    page_df: dataframe that has page level data
    dpc_df: dataframe that has the document level data
    column: teh column on the dataframe that a valeu is going to be set.
    """
    for ind in doc_df.index:
        entities = set()
        for entityList in list(
            page_df.loc[page_df["doc"] == doc_df["File"][ind]][column]
        ):
            try:
                for entity in entityList:
                    entities.add(entity)
            except Exception:
                continue
        if entities:
            doc_df.at[ind, column] = sorted(entities)
    return doc_df


def set_text_doc(page_df, doc_df, column):
    for ind in doc_df.index:
        text = ""
        for textVal in list(
            page_df.loc[page_df["doc"] == doc_df["File"][ind]][column]
        ):
            text += textVal
        doc_df.at[ind, column] = text
    return doc_df


def highlight_text(df, column, col, list_):

    simialrrows = []

    for ind, sentence in zip(
        df.loc[df[col].notnull()].index, df.loc[df[col].notnull()][col]
    ):
        for name in set(list_) & set(sentence):
            sentence = str(sentence).replace(
                name, f'<span style="color: red;">{name}</span>'
            )

            simialrrows.append(df.iloc[ind])

        df[column][ind] = sentence

    return simialrrows


def uncollapse_images(
    collected_hash,
    columns=["Path", "File", "Date", "Name", "Text", "CaseNumber", "Label"],
):
    extracted_info_df = pd.DataFrame(columns=columns)
    for idx, images in zip(collected_hash.index, collected_hash.images):
        for image in images:
            df_dic = dict()
            for k in ["Date", "Name", "Text", "CaseNumber", "Label"]:
                df_dic[k] = collected_hash[k][idx]
            # print(df_dic)
            df_dic["Path"] = image
            df_dic["File"] = image.split("/")[-2]
            # print(df_dic)
            extracted_info_df = extracted_info_df.append(df_dic, ignore_index=True)
    return extracted_info_df


def doc_level_data(extracted_df):
    columns = extracted_df.columns.tolist()
    columns.remove("Path")

    doc_data = pd.DataFrame(columns=columns)

    doc_data["File"] = list(set(extracted_df.doc))

    doc_data
    for column in doc_data.columns:
        if column != "File":
            set_entities_doc(extracted_df, doc_data, column)
    set_text_doc(extracted_df, doc_data, "Text")
    doc_data.to_csv(open("doc_data.csv", "w"))


def highlight_(df, column, col, list_):
    for ind, sentence in zip(
        df.loc[df[col].notnull()].index, df.loc[df[col].notnull()][col]
    ):
        #         print(len(set(list_) & set(sentence)))
        #         if col=='Names':
        #             df[count][ind]=len(set(sentence).intersection(set(list_)))
        for name in set(list_) & set(sentence):
            sentence = str(sentence).replace(
                name, f'<span style="color: red;">{name}</span>'
            )

        #     display(HTML(sentence))
        #     print(doc_data['File'][ind])
        df[column][ind] = sentence
    return df


def find_matches(df, caseentities):
    if set(caseentities) & set(df):
        return len(set(caseentities) & set(df))
    else:
        return 0


# labels=['image', 'other', 'narrative',  'handwriting', 'interview']


def move_images_to_cat(
    path,
    extracted_info_df,
    labels=["image", "other", "narrative", "handwriting", "interview"],
):
    for root, dir, files in os.walk(path):
        for lable in labels:
            os.mkdir(root + "/" + lable)
        for file in files:
            try:
                target_label = extracted_info_df.loc[
                    extracted_info_df["Path"] == root + "/" + file
                ]["label"].values[0]
                shutil.move(root + "/" + file, root + "/" + target_label + "/" + file)
            except Exception:
                print(root + "/" + file)


def read_from_csv(
    path="doc_data.csv", labels=["Name", "Date", "CaseNumber", "Label"], drop=True
):

    doc_data = pd.read_csv(open(path, "r"))
    if drop:
        doc_data = doc_data.drop(["Unnamed: 0"], 1)
    for label in labels:
        for idx, val in zip(
            doc_data.loc[doc_data[label].notnull()].index,
            doc_data.loc[doc_data[label].notnull()][label],
        ):
            entity = val.split("', '")
            entity = [re.sub(r"[\[\]']", "", x) for x in entity]
            doc_data.at[idx, label] = entity
    return doc_data


def get_page_num(page_df, doc_df):
    doc_df["PageRange"] = np.nan
    files = list(set(doc_df["File"]))
    page_doc = {}
    for file in files:
        pages = page_df.loc[page_df["FileSplit"] == file].sort_values(by="Path")["Path"]
        if len(pages) == 0:
            # print(file)
            pages = page_df.loc[page_df["File"] == file].sort_values(by="Path")["Path"]
        # indexes=pages.index
        # if len(pages)>0:
        try:
            first_page, last_page = (
                pages.tolist()[0].split("/")[-1].split("-")[-1],
                pages.tolist()[-1].split("/")[-1].split("-")[-1],
            )
            first_page, last_page = re.sub(".jpg", "", first_page).lstrip("0"), re.sub(
                ".jpg", "", last_page
            ).lstrip("0")
            page_doc[file] = first_page + "-" + last_page
        except Exception:
            print(file)
        # page_df.loc[indexes,'PageRange']=[first_page + '-' +last_page]*len(indexes)
    return page_doc


def set_doc_data(extracted_info_df):
    columns = ["CaseNumber", "Name", "Date", "File"]

    doc_data = pd.DataFrame(columns=columns)

    doc_data["File"] = list(set(extracted_info_df.doc))
    for column in doc_data.columns:
        if column not in ["File", "OrignalFile"]:
            set_entities_doc(extracted_info_df, doc_data, column)
    set_text_doc(extracted_info_df, doc_data, "Text")
    return doc_data


def split_large_files(doc_data, extracted_info_df, minNum=4):
    large_files = []
    for indx in doc_data.loc[doc_data["CaseNumber"].notnull()].index:

        if len(doc_data["CaseNumber"][indx]) > minNum:
            large_files.append(
                extracted_info_df.loc[
                    extracted_info_df["File"] == doc_data["File"][indx]
                ]
            )
            doc_data = doc_data.drop(indx)
    large_files_df = pd.concat(large_files)
    large_files_df["File"] = large_files_df["File"] + large_files_df["DocClass"].astype(
        str
    )

    append_df = set_doc_data(large_files_df)

    doc_data = doc_data.append(append_df)
    return doc_data


def name_and_date_match(doc_data, caseList, ID_column="CaseNumber"):
    nocase = doc_data.loc[doc_data[ID_column].isna()].loc[doc_data["Name"].notnull()]
    name_list = [x for x in nocase.index if len(nocase["Name"][x]) > 10]
    data = {}
    for ind in name_list:
        match = {}
        for casefiles in caseList:
            # if len(nocase['Name'][ind])>10:
            match[casefiles] = find_matches(nocase["Name"][ind], casefiles.names) / len(
                nocase["Name"][ind]
            )
            if not nocase["Date"].isnull()[ind]:
                # print(nocase['Date'][ind])
                match[casefiles] += find_matches(
                    nocase["Date"][ind], casefiles.dates
                ) / len(nocase["Date"][ind])
        data[ind] = dict(sorted(match.items(), key=lambda item: item[1]))
    return data


def get_eligible_rows(casefile, data, t=0.9):
    idx = []
    for k, v in data.items():
        if v[casefile] == max(v.values()) and v[casefile] >= t and v[casefile] != 1:
            print(v[casefile])
            idx.append(k)
    return idx


def get_eligible_case(caseList, data, last_case=0, t=1.0):
    print(last_case)
    for case in caseList[last_case:]:
        indexes = get_eligible_rows(case, data, t)
        last_case += 1
        # print(last_case)
        if len(indexes) > 0:
            # print(last_case)
            return case, indexes, last_case
        else:
            pass
    print("No eligible rows")
    return None, None, 0


def output_organized_cases(caseList):
    i = 1
    for case in caseList:
        print("Case ", i)
        print("Case Numbers in case: ", case.ids)
        print()
        for file in sorted(case.files):
            file_name, page_range = file.split("  ")
            file_name = re.sub("__", "/", file_name)
            file_name = file_name.split(".")
            if file_name[-1] != "pdf":
                file_name = file_name[:-1]
                file_name[-1] = file_name[-1][:3]
            file_name = ".".join(file_name)
            print("     PATH: ", file_name)
            print("     FILE NAME: ", file_name.split("/")[-1])
            print("     PAGE RANGE: ", page_range, end="\n")
        i += 1
        print("*" * 90, end="\n\n")


def split_by_label(extracted_info_df, label="form"):
    files = set(extracted_info_df.doc.tolist())
    extracted_info_df["FileSplit"] = np.nan
    for file in files:
        i = 0
        same_doc = extracted_info_df.sort_values(by="Path").loc[
            extracted_info_df["File"] == file
        ]
        for idx, page in zip(same_doc.index, same_doc["Path"]):
            if extracted_info_df["Label"][idx] == label:
                i += 1
            extracted_info_df.at[idx, "FileSplit"] = (
                extracted_info_df["File"][idx] + str(i) + ".0"
            )
            # print(extracted_info_df['File'][idx]+str(i))
            extracted_info_df.at[idx, "DocClass"] = i

    extracted_info_df = extracted_info_df.sort_values(by="Path")
    return extracted_info_df
