import cv2
from collections import defaultdict
import numpy as np
from IPython.display import IFrame
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import tqdm
import itertools
from exact_duplicate_detection_functions import (
    read_from_csv, HashPages
)
from model_functions import set_up_model, prepare_data_for_model, label_and_vectorize, update_dataframe
from math import ceil
import faiss


def cluster_kmeans(vector, k=2, iter=10, n=4):
    k = ceil(len(vector) / k)
    kmeans = faiss.Kmeans(len(vector[0]), k, niter=iter, verbose=False, nredo=10)
    kmeans.train(vector.cpu().numpy())
    index = faiss.IndexFlatL2(len(vector[0]))
    index.add(vector.cpu().numpy())
    return index.search(kmeans.centroids, n)


def get_indexes_and_vectors_per_cat(labels, cat, outputs):
    indexes = [i for i, x in enumerate(labels) if x == cat]
    vectors = outputs[1:].view(len(labels), -1)[indexes]
    return indexes, vectors


def get_correlation(indexes, encoded_dataset, I, data_df, page_type):
    # pairs=[]
    correlation_df = pd.DataFrame(
        columns=["path_page_0", "path_page_1", "correlation_value", "page_type"]
    )
    for nns in tqdm.tqdm(I):
        possible_pairs = itertools.combinations(np.array(indexes)[nns.tolist()], 2)

        for x in possible_pairs:
            # print(x, 'ny')
            if x[0] != x[1]:  # and (x[1], x[0]) not in [x[0] for x in pairs]:
                img = np.uint8(encoded_dataset["image"][x[0]].permute(1, 2, 0).numpy())
                template = np.uint8(
                    encoded_dataset["image"][x[1]].permute(1, 2, 0).numpy()
                )

                w, h = (224, 224)

                # methods = ['cv.TM_CCOEFF_NORMED']

                method = eval("cv.TM_CCOEFF_NORMED")

                res = cv.matchTemplate(img, template, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

                if (
                    len(
                        correlation_df.loc[
                            correlation_df["path_page_0"]
                            == data_df["representative_page"][x[1]]
                        ].loc[
                            correlation_df["path_page_1"]
                            == data_df["representative_page"][x[0]]
                        ]
                    )
                    == 0
                    and len(
                        correlation_df.loc[
                            correlation_df["path_page_0"]
                            == data_df["representative_page"][x[0]]
                        ].loc[
                            correlation_df["path_page_1"]
                            == data_df["representative_page"][x[1]]
                        ]
                    )
                    == 0
                ):
                    data = {
                        "path_page_0": data_df["representative_page"][x[0]],
                        "path_page_1": data_df["representative_page"][x[1]],
                        "correlation_value": max_val,
                        "page_type": page_type,
                    }
                    correlation_df = correlation_df.append(data, ignore_index=True)

                    # if max_val>t:
                    #     print(x)
                    #     print(data_df['representative_page'][x[0]])
                    #     print(data_df['representative_page'][x[1]])
                    #     # return x
                    #     pairs.append((x, max_val))
    return correlation_df


def plot_pairs_of_pages(
    correlation_df,
    page_types=[],
    sample_size=1,
    sort_acending=False,
    random=False,
    threshold=0.5,
):
    try:
        correlation_df = correlation_df.loc[
            correlation_df["correlation_value"] >= threshold
        ]
        if len(page_types) > 0:
            correlation_df = correlation_df.loc[
                correlation_df["page_type"].isin(page_types)
            ]
        if sort_acending:
            correlation_df = correlation_df.sort_values(by="correlation_value")
        else:
            correlation_df = correlation_df.sort_values(
                by="correlation_value", ascending=False
            )
        if random:
            correlation_df = correlation_df.sample(n=sample_size)
        else:
            correlation_df = correlation_df.head(sample_size)

        print("Showing correlation between images....")
        for path_1, path_2, correlation, page_type in zip(
            correlation_df["path_page_0"],
            correlation_df["path_page_1"],
            correlation_df["correlation_value"],
            correlation_df["page_type"],
        ):
            img = Image.open(path_1).convert("RGB")
            img2 = Image.open(path_2).convert("RGB")
            plt.rcParams["figure.figsize"] = [10, 5]
            plt.rcParams["figure.dpi"] = 100
            plt.subplot(121), plt.imshow(img)
            plt.subplot(122), plt.imshow(img2)
            plt.show()
            print("Correlation ratio between the pages: ", round(correlation, 2))
            print("Type of the correlated pages: ", page_type)
            print("*" * 78)
    except Exception:
        print("Resulting dataframe is: ", correlation_df)
        print(
            "Looks like there are no near duplicates with the given condition.",
            "Consider adjusting your parameters.",
        )


def set_correlation_threshold(
    correlation_df, threshold_by_page_type={"form": 0.09, "image": 0.5}
):
    correlation_cutoff = []
    global thresholds
    thresholds = threshold_by_page_type
    for page_type, cuttoff in threshold_by_page_type.items():
        page_type_df = correlation_df.loc[correlation_df["page_type"] == page_type]
        correlation_cutoff.append(
            page_type_df.loc[page_type_df["correlation_value"] >= cuttoff]
        )

    #         correlation_cutoff.append(page_type_df)
    return pd.concat(correlation_cutoff)


def ClassifyAndGetPairCorrelation(dataset_path, list_of_page_type, from_backup=False):
    if not from_backup:
        print('Hashing pages....')
        _, reduced_df=HashPages(dataset_path)
        
        data, encoded_data= prepare_data_for_model(reduced_df)


        # set up the model for running the data
        model=set_up_model()

        print('Labeling page types...')
        vector_represtenations, page_types=label_and_vectorize(encoded_data, model)

        reduced_df=update_dataframe(reduced_df, page_types)
        print('Updated and saved dataframe with page labels...')

        correlation_df=[]
        for type_of_page in list_of_page_type:
            page_type_indexes, page_type_vectors=get_indexes_and_vectors_per_cat(page_types, type_of_page, vector_represtenations)
            if len(page_type_indexes)>0:
                print('Clustering pages and calculating correlation between page pairs...')
                #print(page_type_vectors)
                D, I= cluster_kmeans(page_type_vectors)
                correlation_df.append(get_correlation(page_type_indexes, encoded_data,  I, data, type_of_page))
            else:
                print('There are no pages that have '+ type_of_page+' type.')
        correlation_df= pd.concat(correlation_df)
        correlation_df.to_csv(open('./correlation_df.csv','w'))
        
        page_number_info={}
        files=[]
        pages=[]
        for dir in os.listdir(os.path.join(dataset_path,'images')):
            files.append(dir)
            pages.append(len(os.listdir(os.path.join(dataset_path,'images', dir))))
        page_number_info['file_name']=files
        page_number_info['number_of_pages']=pages
        page_info_df=pd.DataFrame.from_dict(page_number_info)
        page_info_df.to_csv(open('page_info.csv','w'))
    else: 
        correlation_df=read_from_csv(os.path.join(dataset_path,'correlation_df.csv'), [])
        correlation_df=correlation_df.drop(columns=['Unnamed: 0'])
    return correlation_df.loc[correlation_df["correlation_value"] > 0.0].loc[
        correlation_df["correlation_value"] < 1.0
    ]


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


def visualize_file_pairs(file_1, file_2, outputPath="./outputs/"):
    path_1 = "__" + file_1
    path_2 = "__" + file_2

    print(path_1)
    print(path_2)
    if [
        path_1.strip("__")
        in file_df.loc[file_df["file_name"] == path_2.strip("__")][
            "near_duplicate_files"
        ]
        .values[0]
        .keys()
    ][0] or [
        path_2.strip("__")
        in file_df.loc[file_df["file_name"] == path_1.strip("__")][
            "near_duplicate_files"
        ]
        .values[0]
        .keys()
    ][
        0
    ]:
        if not os.path.exists(
            outputPath + path_1.split(".")[0] + "_" + path_2.split(".")[0] + "A.pdf"
        ):

            hash_df = read_from_csv("./hash_df.csv")
            path_1pages = set(hash_df.loc[hash_df["file_name"] == path_1]["page_path"])
            path_2pages = set(hash_df.loc[hash_df["file_name"] == path_2]["page_path"])
            if not path_1pages or not path_2pages:
                print(
                    "The file path you provided does not have any PDF pages in it."
                    "Please check the path and try again."
                )
                return [], []

            all_pages_2 = [
                Image.open(x).resize((500, 600), Image.ANTIALIAS)
                for x in sorted(path_2pages)
            ]
            for dups in file_df.loc[file_df["file_name"] == path_1.strip("__")][
                "near_duplicate_info"
            ]:
                for dup, infos in dups.items():
                    for info in infos:
                        if info[0].split("/")[-2] == path_2:
                            change = sorted(path_2pages).index(info[0])
                            all_pages_2[change] = Image.fromarray(
                                color_image(info[0], True)
                            ).resize((200, 300), Image.ANTIALIAS)
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

            all_pages_1 = [
                Image.open(x).resize((500, 600), Image.ANTIALIAS)
                for x in sorted(path_1pages)
            ]
            for pages, dups in zip(
                hash_df.loc[hash_df["file_name"] == path_2]["page_path"],
                file_df.loc[file_df["file_name"] == path_2.strip("__")][
                    "near_duplicate_info"
                ],
            ):
                for dup, infos in dups.items():
                    for info in infos:
                        if info[0].split("/")[-2] == path_1:
                            change = sorted(path_1pages).index(info[0])
                            all_pages_1[change] = Image.fromarray(
                                color_image(info[0], True)
                            ).resize((200, 300), Image.ANTIALIAS)
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

            with open("visualize.html", "w") as f:
                f.write(
                    """<!DOCTYPE html>
<html>
    <body>
        <div>"""
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
                    """</div>
    </body>
</html>"""
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
    else:
        print("The two files do not have near duplicate pages in common.")
        return None, None


def print_near_duplciate_information(filtered_df):
    filtered_df["Doc A"] = filtered_df["path_page_0"].apply(
        lambda x: x.split("/")[-2].split("__")[-1]
    )
    filtered_df["Doc B"] = filtered_df["path_page_1"].apply(
        lambda x: x.split("/")[-2].split("__")[-1]
    )

    filter_reverse = pd.DataFrame()
    filter_reverse["path_page_0"] = filtered_df["path_page_1"]
    filter_reverse["path_page_1"] = filtered_df["path_page_0"]
    filter_reverse["correlation_value"] = filtered_df["correlation_value"]
    filter_reverse["page_type"] = filtered_df["page_type"]
    filter_reverse["Doc A"] = filtered_df["Doc B"]
    filter_reverse["Doc B"] = filtered_df["Doc A"]

    filtered_df = pd.concat([filtered_df, filter_reverse])
    filtered_df.index = [x for x in range(len(filtered_df))]
    # print(filtered_df)
    global file_df
    file_df = pd.DataFrame(
        columns=["file_name", "near_duplicate_info", "near_duplicate_files"]
    )
    file_list = list(set(filtered_df["Doc A"]))
    file_df["file_name"] = file_list

    for ind, file in zip(file_df.index, file_df["file_name"]):
        data = {}
        data[file] = []
        file_data = filtered_df.loc[filtered_df["Doc A"] == file]
        if len(file_data) != 0:
            for val in file_data.values:
                # print(val)
                data[file].append({val[5]: (val[1], round(val[2], 2), val[3])})
                dd = defaultdict(list)
                for d in data[file]:
                    for key, value in d.items():
                        dd[key].append(value)
                file_df["near_duplicate_info"][ind] = dd
                file_df["near_duplicate_files"][ind] = {
                    y[0]: len(y[1]) for y in dd.items()
                }
    total_pairs = 0
    page_number_info = read_from_csv("./page_info.csv", [])
    page_number_info["number_of_pages"] = page_number_info["number_of_pages"].apply(
        lambda x: int(x)
    )

    total_pairs = len(
        [
            [file_df["file_name"][idx]]
            * len(file_df["near_duplicate_files"][idx].items())
            for idx in file_df.index
        ]
    )
    print("*" * 80)
    print("Summary\n")
    print("Total Number of Pairs of Documents with Near Duplciate Pages: ", total_pairs)
    print("Correlation threshold for Near Duplciate Detection \n")
    for key, value in thresholds.items():
        print("\t", key, ":  ", value)
    print("*" * 80)
    clean_pair = []
    for idx in file_df.index:
        for file, (ndf, ndp) in zip(
            [file_df["file_name"][idx]]
            * len(file_df["near_duplicate_files"][idx].items()),
            file_df["near_duplicate_files"][idx].items(),
        ):
            if (file, ndf) not in clean_pair and (ndf, file) not in clean_pair:
                clean_pair.append((file, ndf))
                print("Doc A\n")
                print("\tName:\t", file)
                print(
                    "\tNumber of Pages:\t",
                    page_number_info.loc[page_number_info["file_name"] == "__" + file][
                        "number_of_pages"
                    ].values[0],
                )
                print("\tNumber of Near Duplicates:\t", ndp)

                print("Doc B\n")
                print("\tName:\t", ndf)
                print(
                    "\tNumber of Pages:\t",
                    page_number_info.loc[page_number_info["file_name"] == "__" + ndf][
                        "number_of_pages"
                    ].values[0],
                )
                print("\tNumber of Near Duplicates:\t", ndp)
                print("_" * 100)
            # print((ndf, ndp))
    # print(total_pairs)
    file_df.to_csv(open("./file_df.csv", "w"))
    return file_df
