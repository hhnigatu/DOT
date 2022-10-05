import itertools
import re
import pandas as pd
import numpy as np
import Levenshtein

import dotlibrary.file_organization_functions as fof

class caseFiles:
    ids=set()
    names=set()
    dates=set()
    files=set()
    def __init__(self, similarrows):
        for case in similarrows:
            self.ids=self.ids.union(set(case['CaseNumber']))
            self.files=self.files.union(set([case['File']]))
            try:
                self.names=self.names.union(set(case['Name']))
            except:
                self.names=set()
            try:
                self.dates=self.dates.union(set(case['Date']))
            except:
                self.dates=set()

    def add_case(self, case):
        
        self.files=self.files.union(set([case['File']]))
        try:
            self.ids=self.ids.union(set(case['CaseNumber']))
        except:
            self.ids=self.ids
        try:
            self.names=self.names.union(set(case['Name']))
        except:
            self.names=self.names
        try:
            self.dates=self.dates.union(set(case['Date']))     
        except:
            self.dates=self.dates

class document:
    ids=[]
    dates=[]
    file=''
    names=[]
    edges={}
    def __init__(self, files, ids, names, dates):
        # if pd.isna(casenumbers):
        self.ids=ids
        # else:
        #     self.case_number=[re.sub("'", "", x) for x in casenumbers.strip('[').strip(']').split(',')]
        self.dates=dates
        self.file=files
        self.names=names
        self.edges={}
    def get_truth_val(self, entity):
        try:
            truth_val=any(pd.isna(entity))
            #print('found non array returned ' + str(truth_val))
        except:
            truth_val=pd.isna(entity)
            #print('found array returned ' + str(truth_val))
        return truth_val
    def compare_ids(self, ids1, ids2):
        try:
            year1, number1 = ids1.split('-')
            year2, number2 = ids2.split('-')
            if year1.strip(' ')==year2.strip(' '):
                if number1.strip('0').strip(' ')==number2.strip('0').strip(' '):
                    return True
        
            else:
                return False
        except:
            return False
    def connected_to_plain(self, page):
        #try:
            idslist=self.ids
            pageidslist=page.ids
            
            if ((not self.get_truth_val(self.ids)) and (not self.get_truth_val(page.ids))):
                if all(ids in self.ids for ids in page.ids):
                    # print('casenumber exact match between' + self.tag + ' ' + page.tag)
                    #page.reason.append('casenumber exact match')
                    try:
                        self.edges[page]+=(1,'ids exact match')
                    except:
                        self.edges[page]=(1,'ids exact match')
                # elif(textdistance.levenshtein.normalized_similarity(self.case_number[0], page.case_number[0])>0.8):
                #     self.reason.append('casenumber similarity match')
                #     #print('casenumber similarity match between' + self.tag + ' ' + page.tag)
                #     try:
                #         self.edges[page]+=(round(textdistance.levenshtein.normalized_similarity(self.case_number[0], page.case_number[0]),2),'casenumber similarity match')
                #     except:
                #         self.edges[page]=(round(textdistance.levenshtein.normalized_similarity(self.case_number[0], page.case_number[0]),2),'casenumber similarity match')
                elif (not self.get_truth_val(idslist)) and (not self.get_truth_val(pageidslist)):
                        #self.reason.append('casenumber formating match')
                        # print(self.case_number)
                        # print(page.case_number)
                        pairs=list(itertools.product(self.ids, page.ids))
                        # print(pairs)
                        for pair in pairs:
                            # print(pair[0])
                            # print(pair[1])
                            if self.compare_ids(pair[0],pair[1]):
                                print('ID formating match between' + pair[0] + ' ' + pair[1])
                        
                                try:
                                    self.edges[page]+=(1,'ID formating match')
                                except:
                                    self.edges[page]=(1,'ID formating match')
class connections:
    def __init__(self):
        self.pages=set()
        self.con=set()
        self.edge_list={}
    def add_vertices(self, page):
        self.pages.add(page)
    def add_edges(self, page1, page2):
        page1.connected_to_plain(page2)
        if(page1.edges):
            self.edge_list[page1]=page1.edges
            self.con.add(page1)
    def print_connections(self):
        connected={}
        for page in self.pages:
            if (page.edges):
                edges=set()
                # print(page.file+ ':')
                edges.add(page.file)
                for key in self.edge_list[page]:
                    # print(key.file + ":" + str(self.edge_list[page][key])+'->')
                    edges.add(key.file)
    
#                 print('\n')
                connected[page.file]=edges
        return connected
    def find_connections(self):
        for page in self.pages:
            for otherpages in self.pages:
                if page!=otherpages:
                    self.add_edges(page,otherpages)        

def find_related_files(doc_data, lables=['File', 'CaseNumber', 'Name', 'Date' ]):
    demo=connections()
    for ind in doc_data.index:
        
        singlepage=document(doc_data['File'][ind], doc_data['CaseNumber'][ind],doc_data['Name'][ind],doc_data['Date'][ind])
        demo.add_vertices(singlepage)
    demo.find_connections()
    con=demo.print_connections()
    con=dict(sorted(con.items(), key=lambda item: len(item[1]), reverse=True))
    return con

def display_ID_connections(key, doc_data, ID_column='CaseNumber', drop_columns=['Text','Name']):
    con=find_related_files(doc_data)
    doc_data.sort_values(by=ID_column).drop(drop_columns, axis=1).style.apply(lambda x: ['background: red' if (con[key].intersection(set([x.File])))  else '' for i in x], axis=1)         

def get_similar_ids( doc_data, listOfIds, ind, ID_column='CaseNumber', t=0.95):
    similarity_score={}
    for caselist in doc_data.loc[doc_data[ID_column].notnull()][ID_column]:
        for case in  caselist:
            similarity_score[case]=[(x, Levenshtein.ratio(case, x)) for x in listOfIds]
    similar=set()
    for id in doc_data[ID_column][ind]:
           for sim in [x[0] for x in similarity_score[id] if x[1]> t]:
                similar.add(sim) 
                
    return similar           
            

def create_case_objects(doc_data, listOfCN, index, drop_list):
    doc_data['highlightedcase']=np.nan
    # listOfCN= list(set([x for y in doc_data.loc[doc_data['CaseNumber'].notnull()]['CaseNumber'].tolist() for x in y]))
    ca=get_similar_ids(doc_data, listOfCN, index)
    case=caseFiles(fof.highlight_text(doc_data, 'highlightedcase', 'CaseNumber', ca))
    for file in case.files:
        drop_list=drop_list + doc_data.index[doc_data['File']==file].tolist()
    return case, drop_list

def mantian_connection_list(con, case):
    for k in case.files:
        try:
            del con[k]
        except:
            pass
    
    return con

def add_cases(case, indexes, drop_list, doc_data):
    for idx in indexes:
        case.add_case(doc_data.iloc[idx])
        drop_list.append(idx)
    return case, drop_list

def generate_key(con,doc_data, drop_list):
    for key in con.keys():
        for file in con[key]:
            if (len(doc_data.drop(drop_list).loc[doc_data['File']==file])!=0):
                return key
            else:
                return
def add_to_staged_case(eligible_case, doc_data, indexes, drop_list, remove_list=[]):
    for remove_elt in remove_list:
        indexes.remove(remove_elt)
    eligible_case, drop_list= add_cases(eligible_case, indexes, drop_list, doc_data)
    return eligible_case, drop_list