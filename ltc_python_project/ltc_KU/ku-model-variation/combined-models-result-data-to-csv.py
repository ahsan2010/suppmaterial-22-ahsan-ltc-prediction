#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:39:27 2024

@author: ahsan
"""


import pandas as pd
import os.path
import numpy as np
import pickle

KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_MINUS_EXP = "ku_all_minus_dev_exp"
KU_ALL_MINUS_PREV_EXP = "ku_all_minus_prev_exp"
KU_ALL_MINUS_COMM_EXP = "ku_all_minus_comm_exp"
KU_ALL_MINUS_PROJ_PROF = "ku_all_minus_proj_prof"
KU_ALL_MINUS_PREV_PROJ = "ku_all_minus_prev_proj_prof"

KU_CUR = "ku_present_proj_exp"
KU_CUR_XIN = "ku_present_proj_exp_xin_all"


KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

KU_ALL_MINUS_EXP_AUTO = "ku_all_minus_dev_exp_auto"
KU_ALL_MINUS_PREV_EXP_AUTO = "ku_all_minus_prev_exp_auto"
KU_ALL_MINUS_COMM_EXP_AUTO = "ku_all_minus_comm_exp_auto"
KU_ALL_MINUS_PROJ_PROF_AUTO = "ku_all_minus_proj_prof_auto"
KU_ALL_MINUS_PREV_PROJ_AUTO = "ku_all_minus_prev_proj_prof_auto"

KU_CUR_AUTO = "ku_present_proj_exp_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

KULTC_SUM_AUTO = "kultc_sum_feat_auto"
KULTC_SUM_XIN_AUTO = "kultc_sum_xin_feat_auto"

all_feature_model = [
    KU_DEV_EXP,
    KU_OTHER_EXP,
    KU_COMMUNITY,
    PROJECT_PROF,
    PREV_PROJ_PROF,
    KU_ALL ,
    XIN_FEAT,
    KU_ALL_XIN_ALL,
    KU_ALL_MINUS_EXP,
    KU_ALL_MINUS_PREV_EXP,
    KU_ALL_MINUS_COMM_EXP,
    KU_ALL_MINUS_PROJ_PROF,
    KU_ALL_MINUS_PREV_PROJ,
    KU_CUR,
    KU_CUR_XIN
    ]

auto_model = [
    KU_DEV_EXP_AUTO,
    KU_OTHER_EXP_AUTO,
    KU_COMMUNITY_AUTO,
    PROJECT_PROF_AUTO,
    PREV_PROJ_PROF_AUTO,
    KU_ALL_AUTO,
    XIN_FEAT_AUTO,
    KU_ALL_XIN_ALL_AUTO,
    KU_ALL_MINUS_EXP_AUTO,
    KU_ALL_MINUS_PREV_EXP_AUTO,
    KU_ALL_MINUS_COMM_EXP_AUTO,
    KU_ALL_MINUS_PROJ_PROF_AUTO,
    KU_ALL_MINUS_PREV_PROJ_AUTO,
    KU_CUR_AUTO,
    KU_CUR_XIN_AUTO,
    KU_DEV_EXP_XIN_AUTO
    ]


hyper_parameter_model = [
    KU_ALL_XIN_ALL_AUTO,
    KU_CUR_XIN_AUTO
    ]

rebuttal_model = [KULTC_SUM_AUTO, KULTC_SUM_XIN_AUTO]

hyper_parameter_model_ku_dev_xin=[KU_DEV_EXP_XIN_AUTO]

smote_model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO]

reb_save_model_dir ="/home/local/SAIL/ahsan/LTC_Project/save-model-dec-15/" 
save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 
save_model_dir_dev_xin = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/hyper-parameter-tune-cheaper-model/'

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)


def load_ku_model_variation_result (model_list, boot_id, year_value):
    result_list = []
    # Load All Feature Model
    for model_name in model_list:
        model_path = "{}feature-models/rf_{}_{}_{}.pkl".format(save_model_dir, model_name, boot_id, year_value)
        loaded_model = pickle.load(open(model_path, 'rb'))
        result_data = [model_name, 'rf' ,year_value, boot_id, loaded_model.model_result['auc'], loaded_model.model_result['precision'], loaded_model.model_result['recall'],loaded_model.model_result['fscore']]
        result_list.append(result_data)
    return result_list

def load_rebuttal_ku_model_variation_result (model_list, boot_id, year_value):
    result_list = []
    # Load All Feature Model
    for model_name in model_list:
        model_path = "{}random-forest/rf_{}_{}_{}.pkl".format(reb_save_model_dir, model_name, boot_id, year_value)
        loaded_model = pickle.load(open(model_path, 'rb'))
        result_data = [model_name, 'rf' ,year_value, boot_id, loaded_model.model_result['auc'], loaded_model.model_result['precision'], loaded_model.model_result['recall'],loaded_model.model_result['fscore']]
        result_list.append(result_data)
    return result_list

def load_smote_variation_result (model_list, boot_id, year_value):
    sm_variation = ["rf_somte", "rf_somte_tomek"]
    result_list = []
    # Load All Feature Model
    for model_name in model_list:
        for m in sm_variation:
            model_path = "{}feature-models/{}_{}_{}_{}.pkl".format(save_model_dir, m, model_name, boot_id, year_value)
            loaded_model = pickle.load(open(model_path, 'rb'))
            result_data = [model_name + "_hpt_" + m, m ,year_value, boot_id, loaded_model.model_result['auc'], loaded_model.model_result['precision'], loaded_model.model_result['recall'],loaded_model.model_result['fscore']]
            result_list.append(result_data)
    return result_list


def load_hyper_model_variation(model_list, boot_id, year_value, save_model_dir):
    result_list = []
    algo_model = ['rf', 'dtree', 'knn', 'NB', 'XGB', 'LGBM']
    # Load All Feature Model
    for model_name in model_list:
        for m in algo_model:
            model_path = "{}hyper-parameter-tuning-with-test/{}_{}_{}_{}.pkl".format(save_model_dir, model_name, m, boot_id,year_value)
            loaded_model = pickle.load(open(model_path, 'rb'))
            result_data = [model_name + "_hpt_" + m, m ,year_value, boot_id, loaded_model.model_result['auc'], loaded_model.model_result['precision'], loaded_model.model_result['recall'],loaded_model.model_result['fscore']]
            result_list.append(result_data)
    return result_list
    
def load_ku_model_variation(boot_lim):
    boot_iteration = list(range(1,boot_lim + 1,1))
    full_result = []
    for year_value in [1, 2, 3]:
        print("Loading KU Feature Models: Year: {}".format(year_value))
        # per year data load
        for boot_id in boot_iteration:
            #per boot data load
            result_all_feature_model = load_ku_model_variation_result(all_feature_model, boot_id, year_value)
            result_auto_model = load_ku_model_variation_result(auto_model, boot_id, year_value)
            result_reb_model = load_ku_model_variation_result(rebuttal_model, boot_id, year_value)
            result_hyper_model = load_hyper_model_variation(hyper_parameter_model, boot_id, year_value, save_model_dir)
            result_smote_model = load_smote_variation_result (smote_model_list, boot_id, year_value)
            result_hyper_model_ku_dev_xin = load_hyper_model_variation(hyper_parameter_model_ku_dev_xin, boot_id, year_value, save_model_dir_dev_xin)
            full_result.extend(result_all_feature_model)
            full_result.extend(result_auto_model)
            full_result.extend(result_reb_model)
            full_result.extend(result_hyper_model)
            full_result.extend(result_smote_model)
            full_result.extend(result_hyper_model_ku_dev_xin)
    
    data = pd.DataFrame(full_result)
    data.columns = ['model_name', 'algo', 'year', 'boot_id', 'auc', 'precision', 'recall', 'fscore']
    #data.to_csv("/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/full_model_result.csv", index = False)
    data.to_csv("/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/full_model_result_jan_30_2025.csv", index = False)
    


def main():
    load_ku_model_variation(100)
    print('This is the main function')


if __name__ == "__main__":
    main()
    print("Program finishes successfully.")