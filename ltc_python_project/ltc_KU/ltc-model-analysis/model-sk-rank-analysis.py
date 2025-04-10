#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:09:40 2024

@author: ahsan
"""

import pandas as pd

model_result_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap-dec-15/"
sk_analysis_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/model-sk-analysis/"

KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

KU_EXPERIENCE = "experience_ku"
KU_PROJECT = "project_ku"
CURRENT_PROJ_EXP_KU = "current_proj_exp_ku"
PREV_PROJ_EXP_KU = "prev_proj_exp_ku"

KU_XIN_DEV = "ku_xin_dev_dim"
KU_XIN_REPO = "ku_xin_repo_dim"
KU_XIN_DEV_ACT = "ku_xin_dev_act_dim"
KU_XIN_REPO_ACT = "ku_xin_repo_act_dim"
KU_XIN_COLLAB_NET = "ku_xin_collab_net_xim"
KU_XIN_ALL = "ku_xin_all_dim"

model_ku_xin_comb = [
                     KU_ALL,
                     XIN_FEAT,
                     KU_XIN_DEV,
                     KU_XIN_REPO,
                     KU_XIN_DEV_ACT,
                     KU_XIN_REPO_ACT,
                     KU_XIN_COLLAB_NET,
                     KU_XIN_ALL
                     ]

model_list_all = [KU_ALL,
              XIN_FEAT,
              KU_ALL_XIN_ALL]

model_list_auto_all = [KU_ALL_AUTO,
              XIN_FEAT_AUTO,
              KU_ALL_XIN_ALL_AUTO]


model_list_two = [KU_ALL,
              XIN_FEAT]

model_list_auto_two = [KU_ALL_AUTO,
              XIN_FEAT_AUTO]


model_list_new = [
    KU_ALL,
    KU_EXPERIENCE, 
    KU_PROJECT, 
    CURRENT_PROJ_EXP_KU, 
    PREV_PROJ_EXP_KU]
  
def generate_sk_data_for_model(model_list, auto_string):
    ltc_year_list = [1, 2, 3]
    for year in ltc_year_list:
        model_sk_data = {}
        for model_name in model_list:
            model_sk_data[model_name] = []
        for model_name in model_list:
            file_location = "{}ltc_{}_{}_100.csv".format(model_result_dir, year, model_name)
            data = pd.read_csv(file_location)
            data = data[data['classifier'] == 'RandomForest']
            auc_values = data['auc'].to_list()
            model_sk_data[model_name].extend(auc_values)
        
        
        sk_output_file = "{}sk_model_input_year_{}{}_two.txt".format(sk_analysis_dir,year, auto_string)
        with open(sk_output_file, 'w') as fp:
            for m in model_list:
                auc_value_str = ' '.join([str(x) for x in model_sk_data[m]])
                fp.write(m)
                fp.write(' ')
                fp.write(auc_value_str)
                fp.write('\n')
            fp.close()
            
def generate_sk_data_for_all_model(model_list, auto_string):
    ltc_year_list = [1, 2, 3]
    for year in ltc_year_list:
        model_sk_data = {}
        for model_name in model_list:
            model_sk_data[model_name] = []
        for model_name in model_list:
            file_location = "{}ltc_{}_{}_100.csv".format(model_result_dir, year, model_name)
            data = pd.read_csv(file_location)
            data = data[data['classifier'] == 'RandomForest']
            auc_values = data['auc'].to_list()
            model_sk_data[model_name].extend(auc_values)
        
        
        sk_output_file = "{}sk_model_input_year_{}{}_all.txt".format(sk_analysis_dir,year, auto_string)
        with open(sk_output_file, 'w') as fp:
            for m in model_list:
                auc_value_str = ' '.join([str(x) for x in model_sk_data[m]])
                fp.write(m)
                fp.write(' ')
                fp.write(auc_value_str)
                fp.write('\n')
            fp.close()
            
def generate_sk_data_for_remaining_ku_models(model_list, auto_string):
    ltc_year_list = [1, 2, 3]
    for year in ltc_year_list:
        model_sk_data = {}
        for model_name in model_list:
            model_sk_data[model_name] = []
        for model_name in model_list:
            file_location = "{}ltc_{}_{}_100.csv".format(model_result_dir, year, model_name)
            data = pd.read_csv(file_location)
            data = data[data['classifier'] == 'RandomForest']
            auc_values = data['auc'].to_list()
            model_sk_data[model_name].extend(auc_values)
        
        
        sk_output_file = "{}sk_model_input_year_{}{}_remaining_kus.txt".format(sk_analysis_dir,year, auto_string)
        with open(sk_output_file, 'w') as fp:
            for m in model_list:
                auc_value_str = ' '.join([str(x) for x in model_sk_data[m]])
                fp.write(m+auto_string)
                fp.write(' ')
                fp.write(auc_value_str)
                fp.write('\n')
            fp.close()
            
def generate_sk_data_for_ku_xin_comparison_models(model_list, auto_string):
    ltc_year_list = [1, 2, 3]
    for year in ltc_year_list:
        model_sk_data = {}
        for model_name in model_list:
            model_sk_data[model_name] = []
        for model_name in model_list:
            file_location = "{}ltc_{}_{}{}_100.csv".format(model_result_dir, year,model_name, auto_string)
            data = pd.read_csv(file_location)
            data = data[data['classifier'] == 'RandomForest']
            auc_values = data['auc'].to_list()
            model_sk_data[model_name].extend(auc_values)
        
        sk_output_file = "{}sk_model_input_year_{}{}_ku_xin_combined.txt".format(sk_analysis_dir,year, auto_string)
        with open(sk_output_file, 'w') as fp:
            for m in model_list:
                auc_value_str = ' '.join([str(x) for x in model_sk_data[m]])
                fp.write(m+auto_string)
                fp.write(' ')
                fp.write(auc_value_str)
                fp.write('\n')
            fp.close()


def call_sk_input_for_ku_xin_combined():
    generate_sk_data_for_ku_xin_comparison_models(model_ku_xin_comb, '')
    generate_sk_data_for_ku_xin_comparison_models(model_ku_xin_comb, '_auto')

def main():
    #generate_sk_data_for_model(model_list_two, '')
    #generate_sk_data_for_model(model_list_auto_two, '_auto')
    #generate_sk_data_for_remaining_ku_models(model_list_new, '')
    call_sk_input_for_ku_xin_combined()
    print('Program finishes successfully')

if __name__ == "__main__":
    main()