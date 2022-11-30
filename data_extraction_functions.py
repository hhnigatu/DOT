from PIL import Image
import matplotlib.pyplot as plt
import re
import tqdm
import spacy
import string


from collections import Counter


def clean_case_number(text):
    # print(text)
    text_ = re.findall(r"\s(\d{2}) [-]* (\d{5})\s", text)
    # print(text)
    if text_:
        return ["-".join(x) for x in text_]  # .split()
    else:
        text_ = re.findall(r"\s(\d{2}) [-]* (\d{4})\s", text)
        if text_:
            return ["-".join(x) for x in text_]
        else:
            text_ = re.findall(r"\s(\d{2}) [-]* (\d{2})\s", text)
            if text_:
                return ["-".join(x) for x in text_]
            else:
                text_ = re.findall(r"\s(\d{2}) [-]* (\d{4})\s", text)
                if text_:
                    return ["-".join(x) for x in text_]
                else:
                    return


def get_text_from_bbox(bbox, bbox_list, text):
    text_final = ""
    for bbox_, word in zip(bbox_list, text):
        x1 = max(bbox[0], bbox_[0])
        y1 = max(bbox[1], bbox_[1])
        x2 = min(bbox[2], bbox_[2])
        y2 = min(bbox[3], bbox_[3])
        # if bbox_[0]==486:
        # print(x1, x2, y1, y2)

        inter_area = abs(max((x2 - x1), 0) * max(y2 - y1, 0))
        if inter_area != 0:
            # print(word)
            # word=(word)
            # if word:
            text_final = text_final + " " + word
    # print(text_final)
    return clean_case_number(text_final)


def extract_with_bbox(bbox, encoded_dataset, processor, lable_df):
    possible_entities = []
    for idx in lable_df.index:
        possible_entities.append(
            get_text_from_bbox(
                bbox,
                encoded_dataset[idx]["bbox"].squeeze().tolist(),
                processor.tokenizer.decode(
                    encoded_dataset[idx]["input_ids"].squeeze().tolist()
                ).split(),
            )
        )
    possible_entities = list(
        set([num for page in possible_entities if page is not None for num in page])
    )
    return possible_entities


def get_named_entity(data_frame, entity_type, label="narrative"):
    entity = []
    df_text = data_frame.loc[data_frame["label"] == label]["Text"]

    nlp = spacy.load("en_core_web_sm")
    for ind in tqdm.tqdm(df_text.index):

        doc = nlp(df_text[ind])
        for ent in doc.ents:
            if ent.label_ == entity_type:
                entity.append(ent.text)
    return list(entity)


def clean_entity_list(clean_list, entityList, minLen=3, maxLen=20, maxFreq=10):
    """
    Removes stop words that are in the list and removes entries that do not satisfy the length requirement.

    Parameters:
    clean_list: a list of stop words to be removed from the provided entity list.
    entityList: a list of the detected entities that we would like to clean.
    minLen: minimum allowed length of string.
    maxLen: maximum allowed length of string.

    Returns:
    clean_entitiy: a list of entities that satisfy the cleaning requierments.

    """
    cleaner = spacy.load("en_core_web_sm")

    # mark stop words from the list
    for word in clean_list:
        cleaner.vocab[word.lower()].is_stop = True

    # only retain the entities that are not stop words( not in the clean list)
    clean_entitiy = list(
        [
            " ".join(
                re.sub(
                    "[%s]" % re.escape(string.punctuation),
                    "",
                    re.sub(r"\d", "", str(token).strip()),
                )
                for token in cleaner(entity.lower().strip())
                if not token.is_stop
            )
            for entity in entityList
        ]
    )

    # for each word in each entity, only retain words that are more than a single letter/number
    clean_entitiy = list(
        [
            " ".join(entity for entity in cl_entity.split() if len(entity) > 1)
            for cl_entity in clean_entitiy
        ]
    )

    # for each entity only retain ones that are with in the boundry (minLen, maxLen)
    clean_entitiy = [
        entity
        for entity in clean_entitiy
        if len(entity.strip()) > minLen and len(entity.strip()) < maxLen
    ]

    clean_entitiy = [
        entity[0] for entity in Counter(clean_entitiy).items() if entity[1] > maxFreq
    ]
    print(
        "Removed "
        + str(len(entityList) - len(clean_entitiy))
        + " entites. There are now "
        + str(len(clean_entitiy))
        + " entites."
    )
    return clean_entitiy


def date_cleaner(date_list):
    #     for ind, date_list in zip(doc_data.index, doc_data.loc[doc_data['Dates'].notnull()]['Dates']):
    date_set = set()
    try:
        for date in [
            re.sub(
                "\.",
                "/",
                re.sub(
                    "-", "/", re.sub("'", "", re.sub("\]", "", re.sub("\[", "", x)))
                ),
            )
            for x in date_list
        ]:
            #             if len(date)==8:
            date_clean = date.strip(" ").split(" ")[0]
            #                 print(date_clean)
            date_ = ""
            if re.search("/", date_clean):
                for ix in date_clean.split("/"):
                    if len(ix) == 1:
                        ix = "0" + ix
                    date_ += ix + "/"
                #                     date_=date_.strip('/')
                date_ = date_.strip("/")
                if len(date_) == 8 or len(date_) == 10:
                    date_set.add(date_)
    #         date_list_.append(date_set)
    #         print(sorted(date_set))
    except Exception:
        date_set = set()
    return sorted(date_set)


def get_empty_entity(doc_data, extracted_info_df, drop_list, label):
    entity_dic = {}
    for file in doc_data.drop(drop_list)["File"]:
        df = extracted_info_df.loc[extracted_info_df["File"] == file]
        entity = df.loc[df["Label"] == label]["Path"].tolist()
        if len(entity) > 0:
            entity_dic[file] = entity
    return entity_dic


def check_empty_entity(entity_dic, explored_file):
    for file, entity in entity_dic.items():
        if file not in explored_file:
            print(file)
            print([x.split("/")[-1] for x in entity])
            explored_file.append(file)

            return file, explored_file, [x.split("/")[-1] for x in entity]


def plot_extraction_suggestions(forms, path, file):
    for form_name in forms:
        form_name = path + "/" + file + "/form/" + form_name
        # print(form_name)
        img = Image.open(form_name).convert("RGB")
        plt.imshow(img.crop((0, 50, 1500, 800)))
        plt.show()


def add_value(doc_data, file_name, value, column="CaseNumber"):
    index = doc_data.loc[doc_data["File"] == file_name].index[0]
    try:
        doc_data.at[index, column] += value
    except Exception:
        doc_data.at[index, column] = value
    return doc_data
