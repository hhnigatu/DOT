import itertools
import pandas as pd
import numpy as np
import Levenshtein

import file_organization_functions as fof


class CaseFiles:
    ids = set()
    names = set()
    dates = set()
    files = set()

    def __init__(self, similar_rows):
        for case in similar_rows:
            self.ids = self.ids.union(set(case["CaseNumber"]))
            self.files = self.files.union(set([case["File"]]))
            try:
                self.names = self.names.union(set(case["Name"]))
            except Exception:
                self.names = set()
            try:
                self.dates = self.dates.union(set(case["Date"]))
            except Exception:
                self.dates = set()

    def add_case(self, case):
        self.files = self.files.union(set([case["File"]]))
        try:
            self.ids = self.ids.union(set(case["CaseNumber"]))
        except Exception:
            self.ids = self.ids
        try:
            self.names = self.names.union(set(case["Name"]))
        except Exception:
            self.names = self.names
        try:
            self.dates = self.dates.union(set(case["Date"]))
        except Exception:
            self.dates = self.dates


class Document:
    ids = []
    dates = []
    file = ""
    names = []
    edges = {}

    def __init__(self, files, ids, names, dates):
        self.ids = ids
        self.dates = dates
        self.file = files
        self.names = names
        self.edges = {}

    def get_truth_val(self, entity):
        try:
            truth_val = any(pd.isna(entity))
        except Exception:
            truth_val = pd.isna(entity)
        return truth_val

    def compare_ids(self, ids1, ids2):
        try:
            year1, number1 = ids1.split("-")
            year2, number2 = ids2.split("-")
            if year1.strip(" ") == year2.strip(" "):
                if number1.strip("0").strip(" ") == number2.strip("0").strip(" "):
                    return True

            else:
                return False
        except Exception:
            return False

    def connected_to_plain(self, page):
        # try:
        idslist = self.ids
        pageidslist = page.ids

        if (not self.get_truth_val(self.ids)) and (not self.get_truth_val(page.ids)):
            if all(ids in self.ids for ids in page.ids):
                try:
                    self.edges[page] += (1, "ids exact match")
                except Exception:
                    self.edges[page] = (1, "ids exact match")
            elif (not self.get_truth_val(idslist)) and (
                not self.get_truth_val(pageidslist)
            ):
                pairs = list(itertools.product(self.ids, page.ids))
                for pair in pairs:
                    if self.compare_ids(pair[0], pair[1]):
                        print("ID formating match between" + pair[0] + " " + pair[1])

                        try:
                            self.edges[page] += (1, "ID formating match")
                        except Exception:
                            self.edges[page] = (1, "ID formating match")


class Connections:
    def __init__(self):
        self.pages = set()
        self.con = set()
        self.edge_list = {}

    def add_vertices(self, page):
        self.pages.add(page)

    def add_edges(self, page1, page2):
        page1.connected_to_plain(page2)
        if page1.edges:
            self.edge_list[page1] = page1.edges
            self.con.add(page1)

    def print_connections(self):
        connected = {}
        for page in self.pages:
            if page.edges:
                edges = set()
                edges.add(page.file)
                for key in self.edge_list[page]:
                    edges.add(key.file)
                connected[page.file] = edges
        return connected

    def find_connections(self):
        for page in self.pages:
            for otherpages in self.pages:
                if page != otherpages:
                    self.add_edges(page, otherpages)


def find_related_files(doc_data, lables=["File", "CaseNumber", "Name", "Date"]):
    demo = Connections()
    for ind in doc_data.index:

        singlepage = Document(
            doc_data["File"][ind],
            doc_data["CaseNumber"][ind],
            doc_data["Name"][ind],
            doc_data["Date"][ind],
        )
        demo.add_vertices(singlepage)
    demo.find_connections()
    con = demo.print_connections()
    con = dict(sorted(con.items(), key=lambda item: len(item[1]), reverse=True))
    return con


def display_ID_connections(
    key, doc_data, ID_column="CaseNumber", drop_columns=["Text", "Name"]
):
    con = find_related_files(doc_data)
    doc_data.sort_values(by=ID_column).drop(drop_columns, axis=1).style.apply(
        lambda x: [
            "background: red" if (con[key].intersection(set([x.File]))) else ""
            for i in x
        ],
        axis=1,
    )


def get_similar_ids(doc_data, listOfIds, ind, ID_column="CaseNumber", t=0.95):
    similarity_score = {}
    for caselist in doc_data.loc[doc_data[ID_column].notnull()][ID_column]:
        for case in caselist:
            similarity_score[case] = [
                (x, Levenshtein.ratio(case, x)) for x in listOfIds
            ]
    similar = set()
    for id in doc_data[ID_column][ind]:
        for sim in [x[0] for x in similarity_score[id] if x[1] > t]:
            similar.add(sim)

    return similar


def create_case_objects(doc_data, listOfCN, index, drop_list):
    doc_data["highlightedcase"] = np.nan
    ca = get_similar_ids(doc_data, listOfCN, index)
    case = CaseFiles(fof.highlight_text(doc_data, "highlightedcase", "CaseNumber", ca))
    for file in case.files:
        drop_list = drop_list + doc_data.index[doc_data["File"] == file].tolist()
    return case, drop_list


def mantian_connection_list(con, case):
    for k in case.files:
        try:
            del con[k]
        except Exception:
            pass

    return con


def add_cases(case, indexes, drop_list, doc_data):
    for idx in indexes:
        case.add_case(doc_data.iloc[idx])
        drop_list.append(idx)
    return case, drop_list


def generate_key(con, doc_data, drop_list):
    for key in con.keys():
        for file in con[key]:
            if len(doc_data.drop(drop_list).loc[doc_data["File"] == file]) != 0:
                return key
            else:
                return


def add_to_staged_case(eligible_case, doc_data, indexes, drop_list, remove_list=[]):
    for remove_elt in remove_list:
        indexes.remove(remove_elt)
    eligible_case, drop_list = add_cases(eligible_case, indexes, drop_list, doc_data)
    return eligible_case, drop_list
