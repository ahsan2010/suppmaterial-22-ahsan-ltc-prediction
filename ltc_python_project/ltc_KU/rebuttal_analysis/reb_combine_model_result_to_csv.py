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

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index_30_jan_2025.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/models/" 

KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"

KU_OTHER_EXP_MED_AUTO    = "prev_exp_med_prof_auto"
KU_COMMUNITY_MED_AUTO    = "com_exp_med_prof_auto"
PREV_PROJ_PROF_MED_AUTO  = "prev_proj_med_prof_auto"

KU_OTHER_EXP_SUM_AUTO    = "prev_exp_sum_prof_auto"
KU_COMMUNITY_SUM_AUTO    = "com_exp_sum_prof_auto"
PREV_PROJ_PROF_SUM_AUTO  = "prev_proj_sum_prof_auto"

KU_ALL_MED_AUTO          = "ku_all_dim_med_prof_auto"
KU_ALL_SUM_AUTO          = "ku_all_dim_sum_prof_auto"

KU_ALL_MED_SUM_AUTO = "ku_all_dim_med_sum_prof_auto"

XIN_FEAT_AUTO        = "xin_feature_auto"

KU_ALL_MED_XIN_ALL_AUTO  = "ku_all_mid_xin_all_auto"
KU_ALL_SUM_XIN_ALL_AUTO  = "ku_all_sum_xin_all_auto"

KU_ALL_MED_SUM_XIN_ALL_AUTO = "ku_all_mid_sum_xin_all_auto"

KU_CUR_AUTO = "ku_present_proj_exp_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

KULTC_SUM_AUTO = "kultc_sum_feat_auto"
KULTC_SUM_XIN_AUTO = "kultc_sum_xin_feat_auto"

model_list = [KU_ALL_MED_AUTO, KU_ALL_SUM_AUTO, XIN_FEAT_AUTO,
                  KU_ALL_MED_XIN_ALL_AUTO,KU_ALL_SUM_XIN_ALL_AUTO,
                  KU_ALL_MED_SUM_AUTO,KU_ALL_MED_SUM_XIN_ALL_AUTO,
                  KU_DEV_EXP_XIN_AUTO]

def load_ku_model_variation_result (model_list, boot_id, year_value):
    result_list = []
    # Load All Feature Model
    for model_name in model_list:
        model_path = "{}feature-models/rf_{}_{}_{}.pkl".format(save_model_dir, model_name, boot_id, year_value)
        loaded_model = pickle.load(open(model_path, 'rb'))
        result_data = [model_name, 'rf' ,year_value, boot_id, loaded_model.model_result['auc'], loaded_model.model_result['precision'], loaded_model.model_result['recall'],loaded_model.model_result['fscore']]
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
            result_all_feature_model = load_ku_model_variation_result(model_list, boot_id, year_value)
            full_result.extend(result_all_feature_model)
            
    
    data = pd.DataFrame(full_result)
    data.columns = ['model_name', 'algo', 'year', 'boot_id', 'auc', 'precision', 'recall', 'fscore']
    data.to_csv("/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/results/full_model_result_jan_30_2025_med_sum.csv", index = False)
    


def main():
    load_ku_model_variation(100)
    print('This is the main function')


if __name__ == "__main__":
    main()
    print("Program finishes successfully.")