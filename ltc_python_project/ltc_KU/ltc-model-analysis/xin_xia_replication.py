# -*- coding: utf-8 -*-
import pandas as pd
import sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import random
from datetime import datetime
import imblearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.simplefilter('ignore') #we don't wanna see that
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import csv

data_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/'

git_project_file_name = "{}projects.csv".format(data_path)

csv_reader = None 
file = open(git_project_file_name, mode='r', encoding='utf-8')
csv_reader = csv.reader(file, doublequote=True, escapechar='\\',
                            quotechar="\"", delimiter=',',
                                   skipinitialspace=True)

counter = 0
rows = []
for row in csv_reader:
    counter = counter + 1
    if counter % 100000 == 0:
        print(counter)
    if len(row) > 4:
        rows.append([row[0], row[1], row[2], row[5]])
    else:
         rows.append([row[0], row[1], row[2], ""])
        
    '''if counter > 10:
        break'''

pd_git_project_info = pd.DataFrame(rows)        
pd_git_project_info.columns = ['id','url','owner_id','language']

file_name = "{}data_year_1.csv".format(data_path)
pd_year_one_ltc = pd.read_csv(file_name)
pd_year_one_ltc_java = pd_year_one_ltc[pd_year_one_ltc['language'] == 'Java']

repo_ids = list(pd_year_one_ltc_java['repo_id'].unique())
repo_ids_string = [str(x) for x in repo_ids]
java_projects_repo = pd_git_project_info[pd_git_project_info['id'].isin(repo_ids_string)]



def url_string_format(url_str):
    str_x = url_str.split("/")
    x = "https://github.com/{}/{}".format(str_x[4], str_x[5])
    return x

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}_{}".format(str_x[4], str_x[5])
    return x
    

java_projects_repo['html_url'] = [ url_string_format(x) for x in java_projects_repo['url']]
java_projects_repo['project_name'] = [ url_project_name(x) for x in java_projects_repo['url']]
java_projects_repo.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv', index = False)




