#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:46:54 2023

@author: ahsan
"""
# -*- coding: utf-8 -*-
import pandas as pd
import os.path
import numpy as np


#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/August_05_2022/'
#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/Result/'
#model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension/"
#model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap/"
model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap-dec-15/"



ltc_year_list = [1,2,3]
    
KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

model_list = [KU_DEV_EXP,
              KU_OTHER_EXP,
              KU_COMMUNITY,
              PROJECT_PROF,
              PREV_PROJ_PROF,
              KU_ALL,
              XIN_FEAT,
              KU_ALL_XIN_ALL]

model_list_auto = [KU_DEV_EXP_AUTO,
              KU_OTHER_EXP_AUTO,
              KU_COMMUNITY_AUTO,
              PROJECT_PROF_AUTO,
              PREV_PROJ_PROF_AUTO,
              KU_ALL_AUTO,
              XIN_FEAT_AUTO,
              KU_ALL_XIN_ALL_AUTO]

dictionary_model_file = {
                    'XIN_FEATURES'                  : 'xin_features_boostrap',
                    'XIN_FEATURES_KU_COUNT_NORM'    : 'ku_full_xin_features_boostrap',
                    'XIN_FEATURES_KU_COUNT'         : 'ku_count_xin_features_boostrap',
                    'XIN_FEATURES_KU_NORM'          : 'ku_norm_xin_features_boostrap',
                    'KU_COUNT_NORM'                 : 'ku_full_boostrap',
                    'XIN_SIMPLIFIED'                : 'xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_COUNT'       : 'ku_count_xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_NORM'        : 'ku_norm_xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_COUNT_NORM'  : 'ku_full_xin_simp_boostrap',
                    'KU_COUNT'                      : 'ku_count_boostrap',
                    'KU_NORM'                       : 'ku_norm_boostrap'
                    }


def analyze_model_result():
    pd_list = []
    for model in model_list:
        for ltc in ltc_year_list:
            file_name = '{}ltc_{}_{}_100.csv'.format(model_path, ltc, model)
            pd_data = pd.read_csv(file_name)
            pd_list.append(pd_data)
    pd_model_result = pd.concat(pd_list)
    
    result_median = pd_model_result.groupby(['ltc_year','features','classifier'])['auc'].agg('median')
    pd_result_median = pd.DataFrame(result_median).reset_index()
    #pd_result_median['key'] = pd_result_median['ltc_year'].astype(str) + '-'  + pd_result_median['features'] + '-' + pd_result_median['classifier']
    temp = pd.pivot_table(pd_result_median, index = ['ltc_year','features'], columns = 'classifier', values = 'auc')
    pd_final_result = pd.DataFrame(temp).reset_index().rename_axis(None, axis=1)
    pd_final_result.to_csv('{}model_result_bootstrap_median.csv'.format(model_path,), index = False)
   

def analyze_model_result_Autospearman():
    pd_list = []
    for model in model_list_auto:
        for ltc in ltc_year_list:
            file_name = '{}ltc_{}_{}_100.csv'.format(model_path, ltc, model)
            pd_data = pd.read_csv(file_name)
            pd_list.append(pd_data)
    pd_model_result = pd.concat(pd_list)
    
    result_median = pd_model_result.groupby(['ltc_year','features','classifier'])['auc'].agg('median')
    pd_result_median = pd.DataFrame(result_median).reset_index()
    temp = pd.pivot_table(pd_result_median, index = ['ltc_year','features'], columns = 'classifier', values = 'auc')
    pd_final_result = pd.DataFrame(temp).reset_index().rename_axis(None, axis=1)
    pd_final_result.to_csv('{}model_result_bootstrap_median_auto.csv'.format(model_path,), index = False)
   
def main():
    analyze_model_result()
    analyze_model_result_Autospearman()
    print('Program finishes successfully')

if __name__ == "__main__":
    main()
