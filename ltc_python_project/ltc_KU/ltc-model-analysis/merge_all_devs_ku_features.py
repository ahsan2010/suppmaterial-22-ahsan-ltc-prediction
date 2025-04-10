# -*- coding: utf-8 -*-
import pandas as pd
import os.path
import numpy as np
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


def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x


path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile/withoutNormalized/'

project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
#output_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile/withoutNormalized/full_dev_ku_LTC.csv'


def merge_csv_all_projects_ku(project_path, path, output_path, is_normalized = False):
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    proj_list = list(pd_proj_data['repository'])


    pd_list = []

    for project in proj_list:
        if is_normalized == False:
            file_path = "{}{}_dev_ku_LTC.csv".format(path,project)
        if is_normalized == True:
            file_path = "{}{}_dev_normalized_ku_LTC.csv".format(path,project)

            
        #print(project)
        if os.path.exists(file_path) == False:
            print("Missing {} {}".format(project, is_normalized))
            continue
        pd_file = pd.read_csv(file_path)
        pd_list.append(pd_file)
    
    pd_merged = pd.concat(pd_list)
    pd_merged.to_csv(output_path,index = False)

project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'


#normalized_ku_feature_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/normalized/'
#output_norm_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/normalized/full_normalized_ltc.csv'
#count_ku_feature_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/withoutNormalized/'
#output_count_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/withoutNormalized/full_count_ltc.csv'


normalized_ku_feature_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/normalized/'
output_norm_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/normalized/full_normalized_ltc.csv'
count_ku_feature_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/withoutNormalized/'
output_count_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/withoutNormalized/full_count_ltc.csv'


merge_csv_all_projects_ku(project_path, normalized_ku_feature_path, output_norm_ku_feature_full, True)
merge_csv_all_projects_ku(project_path, count_ku_feature_path, output_count_ku_feature_full, False)

def match_name(name_list, text):
    if len(text.strip()) <= 0:
        return 0
    if text in name_list:
        return 1
    return 0

def match_name_II(name_list, login_name, dev_name):
    if len(login_name.strip()) <= 0 & len(dev_name) <= 0:
        return 0
    for x in name_list:
        if len(x) > 0:
            if login_name == x:
                return 1
            if dev_name == x:
                return 1
    return 0

def update_dev_name(name_list,login_name, dev_name):
    if len(login_name.strip()) <= 0 & len(dev_name.strip()) <= 0:
        return '?'
    for x in name_list:
        if len(x) > 0:
            if (len(login_name.strip()) > 0) & (login_name == x):
                return x
            if (len(dev_name.strip()) > 0) & (dev_name == x):
                return x
    return '?'

def analyze_mapping_process():
    year_value = 1
    contrib_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Info_CSV/contributors.csv'
    pd_contrib = pd.read_csv(contrib_path)
    pd_contrib = pd_contrib.replace(np.nan, '', regex=True)
    
    first_year_ltc_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_1_java.csv'
    pd_first_year_ltc = pd.read_csv(first_year_ltc_path)
    
    java_xin_project_list_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    pd_java_xin_project = pd.read_csv(java_xin_project_list_path)
    pd_java_xin_project.rename(columns = {'id' : 'repo_id'}, inplace = True)

    pd_first_year_ltc = pd.merge(pd_first_year_ltc, pd_java_xin_project, how = 'inner', on = 'repo_id')
    pd_first_year_ltc = pd.merge(pd_first_year_ltc, pd_contrib, how = 'inner', on = 'user_id')
    pd_first_year_ltc.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/xin_data_with_user_project_ltc_{}.csv'.format(year_value), index = False)

def mapped_developer_project_full_data(year_value, output_norm_ku_feature_full, output_count_ku_feature_full, output_file):
    
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    
    
    contrib_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Info_CSV/contributors.csv'
    pd_contrib = pd.read_csv(contrib_path)
    pd_contrib = pd_contrib.replace(np.nan, '', regex=True)


    #ku_features_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile/withoutNormalized/full_dev_ku_LTC.csv'
    pd_normalized = pd.read_csv(output_norm_ku_feature_full)
    pd_normalized = pd_normalized[~pd_normalized.duplicated()]
    pd_count = pd.read_csv(output_count_ku_feature_full)
    pd_count = pd_count[~pd_count.duplicated()]
    col = list(pd_normalized)
    norm_feature_list = ['norm_' + x for x in col[3:32]]
    norm_feature_list = col[0:3] + norm_feature_list + col[32:35]
    pd_normalized.columns = norm_feature_list
    
    pd_ku_feature = pd.merge(pd_count, pd_normalized, how = 'left', on = col[0:3] + col[32:35] )
    dev_name_list = pd_ku_feature['Developer_Name'].tolist()

    pd_contrib['login_name_match'] = [match_name(dev_name_list,x) for x in pd_contrib['login']]
    pd_contrib['dev_name_match'] = [match_name(dev_name_list,x) for x in pd_contrib['dev_name']]
    pd_contrib['dev_name_update'] = [update_dev_name(dev_name_list,x,y) for (x,y) in zip(pd_contrib['login'],pd_contrib['dev_name'])]
    
    pd_contrib['is_match'] = [match_name_II(dev_name_list,x,y) for (x,y) in zip(pd_contrib['login'],pd_contrib['dev_name'])]
    
    #pd_contrib['is_match'] = pd_contrib['login_name_match'] + pd_contrib['dev_name_match']
    #pd_contrib.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/user_map_result_full_ku_year_{}.csv'.format(year_value), index = False)
    pd_contrib.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/user_map_result_full_ku_year_{}.csv'.format(year_value), index = False)


    #pd_contrib_matched[pd_contrib_matched['dev_name_update'] == ''].index
    
    pd_contrib_matched = pd_contrib[pd_contrib['is_match'] > 0]
    pd_contrib_matched = pd_contrib_matched.rename(columns = {'dev_name':'old_dev_name'})
    pd_contrib_matched = pd_contrib_matched.rename(columns = {'dev_name_update':'dev_name'})

    xin_data_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_{}_java_with_gt_login_name.csv'.format(year_value)
    pd_xin_data = pd.read_csv(xin_data_path)
    pd_filtered_data = pd.merge(pd_xin_data,pd_contrib_matched, how = 'inner', on = 'user_id')

    pd_proj_data.columns = ['repo_id','url','owner_id','language','html_url','Project_Name']
    pd_merged_full = pd.merge(pd_filtered_data,pd_proj_data, how = 'inner', on = 'repo_id')


    pd_ku_feature.rename(columns = {'Developer_Name':'dev_name'}, inplace = True)

    pd_merged_full_1 = pd.merge(pd_ku_feature,pd_merged_full, how = 'inner', on = ['dev_name','Project_Name'])
    
    pd_ku_feature.rename(columns = {'dev_name':'login_x'}, inplace = True)
    pd_merged_full_2 = pd.merge(pd_ku_feature,pd_merged_full, how = 'inner', on = ['login_x','Project_Name'])

    pd_full = pd.concat([pd_merged_full_1, pd_merged_full_2])
    pd_full = pd_full.drop_duplicates()
    pd_full.to_csv('{}full_result_ltc_{}.csv'.format(output_file, year_value), index = False)

#output_norm_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/normalized/full_normalized_ltc.csv'
#output_count_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/ku_feature_dev_profile_June_21_2022/withoutNormalized/full_count_ltc.csv'
#output_full_ku_result = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/'


output_norm_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/normalized/full_normalized_ltc.csv'
output_count_ku_feature_full = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/withoutNormalized/full_count_ltc.csv'
output_full_ku_result = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'


mapped_developer_project_full_data(1, output_norm_ku_feature_full, output_count_ku_feature_full, output_full_ku_result)
mapped_developer_project_full_data(2, output_norm_ku_feature_full, output_count_ku_feature_full, output_full_ku_result)
mapped_developer_project_full_data(3, output_norm_ku_feature_full, output_count_ku_feature_full, output_full_ku_result)
