import os
import cv2
import tqdm
import math
import torch
import shutil
import pytesseract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image, ImageFile
from struct import pack
from pyparsing import col
from datasets import Dataset
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import (
    LayoutLMv2FeatureExtractor,
    LayoutLMv2TokenizerFast,
    LayoutLMv2Processor,
    LayoutLMv2ForSequenceClassification,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_data_for_model(collected_hash, div_ratio=0, batched=True, batch_size=1):
    datas = []
    datasets = []
    if div_ratio != 0:
        div = math.floor(len(collected_hash) / div_ratio)
        for i in range(div_ratio):
            if i != div_ratio - 1:
                data = pd.DataFrame(columns=["representative_page"])
                data["representative_page"] = collected_hash[i * div : (i + 1) * div][
                    "representative_page"
                ].tolist()
                dataset = Dataset.from_pandas(data)

            else:
                data = pd.DataFrame(columns=["representative_page"])
                data["representative_page"] = collected_hash[i * div :][
                    "representative_page"
                ].tolist()
                dataset = Dataset.from_pandas(data)
            datas.append(data)
            datasets.append(dataset)
    else:
        datas = pd.DataFrame(columns=["representative_page"])
        datas["representative_page"] = collected_hash["representative_page"].tolist()
        datasets = Dataset.from_pandas(datas)
        print("Encoding pages for model....")
        encoded_data = encode_dataset(datasets, batched=batched, batch_size=batch_size)

    return datas, encoded_data


def set_features(
    features=Features(
        {
            "image": Array3D(dtype="int64", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
        }
    )
):
    return features


def encode_dataset(dataset, batched=True, batch_size=1):
    encoded_dataset = dataset.map(
        preprocess_data,
        remove_columns=dataset.column_names,
        features=set_features(),
        batched=batched,
        batch_size=batch_size,
    )
    encoded_dataset.set_format(type="torch")
    return encoded_dataset


def set_up_processor(path="microsoft/layoutlmv2-base-uncased"):
    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained(path)
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)

    return processor


def set_up_model(path_to_model="./", device="cpu"):

    model = LayoutLMv2ForSequenceClassification.from_pretrained(path_to_model)
    model.to(device)

    return model


def preprocess_data(batch):
    """
    param:
    batch: a batch of paths

    returns
    encoded_inputs: the encoding of the images after running layoutlmv2 processor.

    """
    try:
        images = [
            Image.open(path).convert("RGB") for path in batch["representative_page"]
        ]

        # text.append([get_text(path) for path in batch['representative_page']])
        try:
            with torch.no_grad():
                processor = set_up_processor()
                encoded_inputs = processor(
                    images, padding="max_length", truncation=True
                )
        except OSError as er:
            print(er)
    except FileNotFoundError as err:
        print(err)
    return encoded_inputs


def get_vector_rep(
    data,
    model,
    labels=["image", "other", "narrative", "form", "handwriting", "interview"],
):
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}
    model.eval()
    with torch.no_grad():

        for k, v in data.items():
            data[k] = v.to(model.device)
            # print(data[k].device)
        output = model.layoutlmv2.visual(data["image"])
        logits = model(**data)
        # embeddings.append(outputs)
        prediction = logits.logits.argmax(-1)

    return (output, id2label[prediction.item()])


def label_and_vectorize(encoded_dataset, model, batch_size=1):
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size)
    outputs = torch.zeros((1, 49, 256)).to(model.device)
    label_output = []
    for data in tqdm.tqdm(dataloader):
        representation = get_vector_rep(data, model)

        outputs = torch.cat([outputs, representation[0]], dim=0).squeeze(1)
        label_output.append(representation[1])

    return outputs, label_output


def update_dataframe(collected_hash, lable_outputs):
    collected_hash["Label"] = lable_outputs
    collected_hash["doc"] = collected_hash["page_path"].apply(
        lambda x: x.split("/")[-2]
    )
    collected_hash.index = [x for x in range(len(collected_hash))]
    return collected_hash


def highlight_label_pages(label, collected_hash):
    collected_hash = (
        collected_hash.sort_values(by="page_path")
        .drop(["representative_page", "images"], axis=1)
        .style.apply(
            lambda x: ["background: red" if x.Label == label else "" for i in x], axis=1
        )
    )
    return collected_hash


def get_page_type(labels, collected_hash):
    labled_hash = collected_hash.loc[collected_hash["Label"] == labels]
    return labled_hash


def plot_page_type_data(collected_hash, labels):
    num = []
    for label in labels:
        num.append(
            sum(
                collected_hash.loc[collected_hash["Label"] == label][
                    "number_of_duplicates"
                ]
            )
        )
    plt.bar(labels, num)
    plt.savefig("page_type_stat.png")
    plt.show()


def create_cat_folders(
    path, labels=["image", "other", "narrative", "handwriting", "interview", "form"]
):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            for lable in labels:
                os.mkdir(root + "/" + dir + "/" + lable)
        break


def save_images_in_cat_folders(path, extracted_info_df):
    for root, dirs, files in os.walk(path):
        for file in files:
            try:
                target_label = extracted_info_df.loc[
                    extracted_info_df["Path"] == root + "/" + file
                ]["Label"].values[0]
                shutil.move(root + "/" + file, root + "/" + target_label + "/" + file)
            except:
                break


def save_page_type_data(collected_hash, file_name="collected_hash_data.csv"):
    collected_hash.to_csv(open(file_name, "w"))


def get_label_df(label, collected_hash):
    return collected_hash.loc[collected_hash["Label"] == label]


def plot_sample_cat(indexes, encoded_dataset, size=5):
    # print(indexes)
    for idx in indexes[:size]:
        # print(idx)
        img = np.uint8(encoded_dataset["image"][idx].permute(1, 2, 0).numpy())
        plt.subplot(121), plt.imshow(img)
        plt.show()
