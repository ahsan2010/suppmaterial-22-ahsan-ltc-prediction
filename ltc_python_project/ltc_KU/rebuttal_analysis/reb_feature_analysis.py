#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:22:59 2024

@author: ahsan
"""

import shap
import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
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
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context
from scipy.stats import pearsonr

from lightgbm import LGBMClassifier
import time
import pickle
import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)
from autospearman_kla_python_impl import AutoSpearman
import warnings
warnings.simplefilter('ignore') #we don't wanna see that

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']


sk_input_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-rank-input/'
data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'


KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

# KU feature dimension index
dev_exp_features_ind = list(range(3,32))
cur_proj_features_ind = list(range(94,123))
prev_dev_exp_features_ind = list(range(123,152))
col_exp_features_ind = list(range(152,181))
prev_proj_features_ind = list(range(181,210))

#Xin feature dimension index
xin_dev_prof = list(range(33,40))
xin_repo_prof = list(range(40,59))
xin_month_repo_prof = list(range(59,77))
xin_dev_act_prof = list(range(77,89))
xin_collab_prof = list(range(89,94))

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)
shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/shap/"

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 

full_model = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO, KU_CUR_XIN_AUTO]

ku_dimension_list = [
    'KU_DEV_EXP',
    'KU_CUR_PROJ',
    'KU_DEV_PREV_EXP',
    'KU_COLLAB_EXP',
    'KU_PREV_PROJ'
    ]

xin_dimension_list = [
    'BAO_DEV_PPROF',
    'BAO_REPO_PROF',
    'BAO_DEV_ACT',
    'BAO_REPO_ACT',
    'BAO_COLLAB_NET'
    ]


def map_feature_to_dimension(pd_feature,model_dimension_list, map_feature_dimension):
    feature_list = list(pd_feature['feature_name'])
    dimension_list = []
    for f in feature_list:
        dimension_list.append(map_feature_dimension[f])
    return dimension_list


def generate_group_shap_values(load_shap_value,dimension_order, feature_dimension_list, feature_list):

    group_shap_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))
    group_data_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))

    for i, group_name in enumerate(dimension_order):
        
        features = feature_dimension_list[group_name]
        
        print('i={} {}'.format(i,group_name))
        features = [f for f in features if f in feature_list]
        indices = [feature_list.index(f) for f in features]  
        group_shap = load_shap_value.values[:, indices] 
        
        group_shap_values[:, i] = load_shap_value.values[:, indices].sum(axis=1)
        group_data_values[:, i] = load_shap_value.data[:, indices].sum(axis=1)
    
    return group_shap_values, group_data_values 

def generate_data_frame_from_group_shap(dimension_order,group_shap_values,group_data_values):
    pd_list = []
    for i, group_name in enumerate(dimension_order):
        d = pd.DataFrame({'shap_value' : group_shap_values[:,i],
                          'data_value' : group_data_values[:, i],
                          'dim' : dimension_order[i]})
        pd_list.append(d)
    pd_final = pd.concat(pd_list)
    return pd_final

def top_bottom_analysis(load_shap_value, feature_dimension_list, dimension_order, year_value, model_name, map_feature_dimension):
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/sing_model_pred_{}.csv'.format(year_value)
    pd_model_pred = pd.read_csv(f)
    pd_model_pred_sorted = pd_model_pred.sort_values(by='predict_prob', ascending=False)
    
    num_rows = len(pd_model_pred_sorted)
    ratio = 25
    top_ratio_count = int(num_rows * (ratio/100.0))
    bottom_ratio_count = int(num_rows * (ratio/100.0))
    
    # Select top 25% of the DataFrame
    top_ratio_df = pd_model_pred_sorted.head(top_ratio_count)
    top_ratio_df['pos'] = 'top'
    
    # Select bottom 25% of the DataFrame
    bottom_ratio_df = pd_model_pred_sorted.tail(bottom_ratio_count)
    bottom_ratio_df['pos'] = 'bottom'
    

    feature_list = load_shap_value.feature_names
    #load_shap_value.feature_names = [map_feature_dimension[f]+":"+ f for f in feature_list]
    
    
    load_shap_value_top_ratio = load_shap_value[top_ratio_df.id.to_list()]
    load_shap_value_bottom_ratio = load_shap_value[bottom_ratio_df.id.to_list()]
    
    
    group_shap_values_top, group_data_values_top = generate_group_shap_values(load_shap_value_top_ratio, dimension_order, feature_dimension_list, feature_list)
    group_shap_values_bottom, group_data_values_bottom = generate_group_shap_values(load_shap_value_bottom_ratio, dimension_order, feature_dimension_list, feature_list)

    pd_top_group = generate_data_frame_from_group_shap(dimension_order, group_shap_values_top, group_data_values_top)
    pd_bottom_group = generate_data_frame_from_group_shap(dimension_order, group_shap_values_bottom, group_data_values_bottom)
    
    f1 = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/top_bottom_analysis/top_ratio_{}_{}.csv'.format(ratio,year_value)
    f2 = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/top_bottom_analysis/bottom_ratio_{}_{}.csv'.format(ratio,year_value)
    
    pd_top_group.to_csv(f1, index=False)
    pd_bottom_group.to_csv(f2, index=False)


def correlation_analysis_shap_values(load_shap_value, feature_dimension_list, dimension_order, year_value, model_name, map_feature_dimension):
    feature_list = load_shap_value.feature_names
    load_shap_value.feature_names = [map_feature_dimension[f]+":"+ f for f in feature_list]
    group_shap_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))
    group_data_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))
    
    group_shap_values, group_data_values = generate_group_shap_values(load_shap_value, dimension_order, feature_dimension_list, feature_list)
    
    cor_result = []
    for i, group_name in enumerate(dimension_order):
        pearson_corr, p_value = pearsonr(group_data_values[:, i], group_shap_values[:,i])
        cor_result.append([dimension_order[i], pearson_corr, p_value])
        
    pd_correlation = pd.DataFrame(cor_result)
    pd_correlation.columns = ['dim','cor','p_value']
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/full_corr_vallue_shap_LTC_{}.csv'.format(year_value)
    pd_correlation.to_csv(f, index=False)
    
def generate_singel_boot_feature_study(load_shap_value, feature_dimension_list, dimension_order, year_value, model_name, map_feature_dimension):
    
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/sing_model_pred_{}.csv'.format(year_value)
    pd_model_pred = pd.read_csv(f)
    pd_model_pred_sorted = pd_model_pred.sort_values(by='predict_prob', ascending=False)
    
    
    pred_index = pd_model_pred['predicted'].to_list()
    indices_with_ones = [index for index, value in enumerate(pred_index) if value == 1]


    feature_list = load_shap_value.feature_names
    load_shap_value.feature_names = [map_feature_dimension[f]+":"+ f for f in feature_list]
    
    temp  = load_shap_value
    
    #load_shap_value = load_shap_value[indices_with_ones]
    
    #id = (load_shap_value.values > -0.1) & (load_shap_value.values < 0)
    #load_shap_value.values[id] = 0.0
    
    #pos_counts = np.sum(load_shap_value.values > 0, axis=1)
    #neg_counts = np.sum(load_shap_value.values < -0.005, axis=1)
    
    group_shap_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))
    group_data_values = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))

    result_direction = []
    
    pos_shap = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))
    neg_shap = np.zeros((len(load_shap_value.values), len(feature_dimension_list)))

    for i, group_name in enumerate(dimension_order):
        
        features = feature_dimension_list[group_name]
        
        print('i={} {}'.format(i,group_name))
        features = [f for f in features if f in feature_list]
        indices = [feature_list.index(f) for f in features]  
        group_shap = load_shap_value.values[:, indices] 
        
        pos_shap[:, i] = np.sum(group_shap * (group_shap > 0), axis=1)
        neg_shap[:, i] = np.sum(group_shap * (group_shap < 0), axis=1)
        
        group_shap_values[:, i] = load_shap_value.values[:, indices].sum(axis=1)
        group_data_values[:, i] = load_shap_value.data[:, indices].sum(axis=1)
    
    ratio = np.divide(pos_shap, np.abs(neg_shap), where=(neg_shap != 0))
    for i, group_name in enumerate(dimension_order):
        positive = np.sum(group_shap_values[:,i] > 0)
        negative = np.sum(group_shap_values[:,i] < 0)
        result_direction.append([year_value, group_name, positive, negative, positive/len(group_shap_values[:,i]), negative/len(group_shap_values[:,i])])
    
    group_shap_explanation = shap.Explanation(
        values=group_shap_values,
        data=group_data_values,  # Include actual data values
        feature_names=dimension_order
    ) 
    
    pd_list = []
    cor_result = []
    for i, group_name in enumerate(dimension_order):
        d = pd.DataFrame({'shap_value' : group_shap_values[:,i],
                          'data_value' : group_data_values[:, i],
                          'dim' : dimension_order[i]})
        pearson_corr, p_value = pearsonr(group_data_values[:, i], group_shap_values[:,i])
        cor_result.append([dimension_order[i], pearson_corr, p_value])
        pd_list.append(d)
        
    pd_correlation = pd.DataFrame(cor_result)
    pd_correlation.columns = ['dim','cor','p_value']
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/corr_vallue_shap_LTC_{}.csv'.format(year_value)
    #pd_correlation.to_csv(f, index=False)
    
    pd_final = pd.concat(pd_list)
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/full_shap_data_value_LTC_{}.csv'.format(year_value)
    pd_final.to_csv(f, index=False)
    mean_shap_values = np.mean(load_shap_value.values, axis=0)
    shap_summary = pd.DataFrame({'Feature': load_shap_value.feature_names, 'Mean_SHAP': mean_shap_values})

    positive_features = shap_summary[shap_summary['Mean_SHAP'] > 0]

    # Sort by impact (descending)
    positive_features = positive_features.sort_values(by='Mean_SHAP', ascending=False)


    col2num = {col: i for i, col in enumerate(dimension_order)}
    
    order = [0,1,2,3,4]
    #shap.summary_plot(load_shap_value[1], load_shap_value.data)
    shap.plots.beeswarm(group_shap_explanation, order = order , show=False)
    #shap.plots.beeswarm(load_shap_value, show=False)
    fig_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/feature_importance/'
    
    #fig_path = "{}{}-{}.pdf".format(fig_dir, model_name, year_value)
    fig_path = "{}{}-{}_postivie_case.pdf".format(fig_dir, model_name, year_value)
    #fig_path = "{}{}-{}_feature_wise.png".format(fig_dir, model_name, year_value)
    
    #plt.savefig(fig_path, format='pdf', dpi=1000, bbox_inches='tight')
   
    #plt.savefig(fig_path, format='pdf', dpi=1000, bbox_inches='tight')
    #plt.show()
    #plt.close()
    
    return feature_list, group_shap_values, group_data_values, result_direction

def get_feature_names(col_list):
    dev_exp_ku_features = [col_list[i] for i in dev_exp_features_ind] 
    cur_proj_ku_features = [col_list[i] for i in cur_proj_features_ind]
    prev_dev_exp_features = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features = [col_list[i] for i in prev_proj_features_ind]
    
    xin_dev_prof_features = [col_list[i] for i in xin_dev_prof]
    xin_repo_prof_features = [col_list[i] for i in xin_repo_prof]
    xin_month_repo_prof_features = [col_list[i] for i in xin_month_repo_prof]
    xin_dev_act_prof_features = [col_list[i] for i in xin_dev_act_prof]
    xin_collab_prof_features = [col_list[i] for i in xin_collab_prof]

    ku_feature_dimension_list ={
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features
        }
    
    xin_feature_dimension_list = {
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
    feature_dimension_list = {
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features,
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
    map_feature_dimension = {}
    for key,value in feature_dimension_list.items():
        f_list = value
        for f in f_list:
            map_feature_dimension[f] = key
    
    return ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension

def shap_top_bottom_analysis(year_value):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    feature_dimension_list, xin_feature_dimension_list, full_feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    dimension_order =           ['KU_DEV_EXP',
                              'KU_DEV_PREV_EXP',
                              'KU_COLLAB_EXP',
                              'KU_CUR_PROJ',
                              'KU_PREV_PROJ']
    for model_name in [KU_ALL_AUTO]:
        shap_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,model_name,year_value)
        load_shap_value = pickle.load(open(shap_file, 'rb'))
        top_bottom_analysis(load_shap_value, feature_dimension_list, dimension_order, year_value, model_name, map_feature_dimension)
        

result_direction_final = []
def shap_featue_analysis(year_value):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    feature_dimension_list, xin_feature_dimension_list, full_feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    dimension_order =           ['KU_DEV_EXP',
                              'KU_DEV_PREV_EXP',
                              'KU_COLLAB_EXP',
                              'KU_CUR_PROJ',
                              'KU_PREV_PROJ']
    
    
    
    for m in [KU_ALL_AUTO]:
        shap_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,m,year_value)
        load_shap_value = pickle.load(open(shap_file, 'rb'))
        feature_list, group_shap_values, group_data_values, result_direction = generate_singel_boot_feature_study(load_shap_value, feature_dimension_list, dimension_order, year_value, "KULTC", map_feature_dimension)
        result_direction_final.extend(result_direction)
        print(result_direction_final)
        print("[Finish] file = {} year = {}".format(m, year_value))

def shap_correlation_analysis(year_value):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    feature_dimension_list, xin_feature_dimension_list, full_feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    dimension_order =           ['KU_DEV_EXP',
                              'KU_DEV_PREV_EXP',
                              'KU_COLLAB_EXP',
                              'KU_CUR_PROJ',
                              'KU_PREV_PROJ']
    for model_name in [KU_ALL_AUTO]:
        shap_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,model_name,year_value)
        load_shap_value = pickle.load(open(shap_file, 'rb'))
        correlation_analysis_shap_values(load_shap_value, feature_dimension_list, dimension_order, year_value, model_name, map_feature_dimension)
        
        
def shap_feature_analysis():
    for year_value in [1,2,3]:
        shap_featue_analysis(year_value)
        
    f = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/feature_importance/feature_impor_dim_direction.csv'
    result_direction_final_pd = pd.DataFrame(result_direction_final)
    result_direction_final_pd.columns = ['LTC', 'dimension', 'pos','neg','per_pos', 'per_neg']
    #result_direction_final_pd.to_csv(f,index = False)


def main_top_bottom_analysis():
    for year_value in [1,2,3]:
        shap_top_bottom_analysis(year_value)
    
def main_correlation_analysis():
    for year_value in [1,2,3]:
        shap_correlation_analysis(year_value)

def main():
    shap_feature_analysis()
    #main_top_bottom_analysis()
    #main_correlation_analysis()
    print("main function..")


if __name__ == "__main__":
    main()
    print("Program finishes successfully.")