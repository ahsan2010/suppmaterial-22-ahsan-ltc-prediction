#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:25:09 2024

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
    KU_CUR_XIN_AUTO
    ]

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

hyper_parameter_model = [
    KU_ALL_XIN_ALL_AUTO,
    KU_CUR_XIN_AUTO
    ]

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)

model_result_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/'
sk_rank_input_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank_input/'

def generate_sk_data_for_remaining_ku_models(model_list, model_name_list, file_name):
    ltc_year_list = [1, 2, 3]
    file_location = "{}full_model_result.csv".format(model_result_dir)
    full_result_data = pd.read_csv(file_location)        
    for year in ltc_year_list:
        year_result_data = full_result_data[full_result_data['year'] == year]
        model_sk_data = {}
        for model_name in model_list:
            model_sk_data[model_name] = []
        for model_name in model_list:
            
            data = year_result_data[year_result_data['model_name'] == model_name]
            #print(f"M: {model_name} SHAPE: {data.shape}")
            auc_values = data['auc'].to_list()
            model_sk_data[model_name].extend(auc_values)
        
        sk_output_file = "{}sk_model_input_year_{}_{}.txt".format(sk_rank_input_dir, file_name, year)
        with open(sk_output_file, 'w') as fp:
            for m in model_list:
                auc_value_str = ' '.join([str(x) for x in model_sk_data[m]])
                fp.write(m)
                fp.write(' ')
                fp.write(auc_value_str)
                fp.write('\n')
            fp.close()


def smote_hyper_data():
    sm_variation = ["rf_somte", "rf_somte_tomek"]
    model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO,KU_ALL_XIN_ALL_AUTO]
    sm_model_list = []
    for m in model_list:
        for s in sm_variation:
            sm_model_list.append(m + "_hpt_" + s)
    
    sm_model_list = model_list + sm_model_list
    
    sm_model_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KU_ALL_XIN_ALL_AUTO: "KULTC+BAOLTC",
        "ku_all_dim_prof_auto_hpt_rf_somte" : "KULTC_SM",
        "ku_all_dim_prof_auto_hpt_rf_somte_tomek" : "KULTC_SMT",
        'xin_feature_auto_hpt_rf_somte' : "BAOLTC_SM",
        'xin_feature_auto_hpt_rf_somte_tomek' : "BAOLTC_SMT",
        'ku_all_xin_all_auto_hpt_rf_somte' : "KULTC+BAOLTC-SM",
        'ku_all_xin_all_auto_hpt_rf_somte_tomek' : "KULTC+BAOLTC-SMT"
        }
    sm_file_name = "sm_analysis"
    generate_sk_data_for_remaining_ku_models(sm_model_list, sm_model_name, sm_file_name)
    

def main():
    #load_ku_model_variation(100)
    
    rq1_model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO]
    rq1_model_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC"
        }
    rq1_file_name = "rq1"
    generate_sk_data_for_remaining_ku_models(rq1_model_list, rq1_model_name, rq1_file_name)
    
    rq2_model_list = [KU_ALL_AUTO, 
                      KU_ALL_MINUS_EXP_AUTO,
                      KU_ALL_MINUS_PREV_EXP_AUTO,
                      KU_ALL_MINUS_COMM_EXP_AUTO,
                      KU_ALL_MINUS_PROJ_PROF_AUTO,
                      KU_ALL_MINUS_PREV_PROJ_AUTO]
    rq2_model_name = {
        KU_ALL_AUTO: "KULTC",
        KU_ALL_MINUS_EXP_AUTO: "KULTC-DEV_CUR_EXP",
        KU_ALL_MINUS_PREV_EXP_AUTO: "KULTC-DEV_PREV_EXP",
        KU_ALL_MINUS_COMM_EXP_AUTO : "KULTC-COLLAB_EXP",
        KU_ALL_MINUS_PROJ_PROF_AUTO: "KULTC-CUR_PROJ_PROF",
        KU_ALL_MINUS_PREV_PROJ_AUTO: "KULTC-PREV_PROJ_PROF"
        }
    rq2_file_name = "rq2"
    generate_sk_data_for_remaining_ku_models(rq2_model_list, rq2_model_name, rq2_file_name)
    
    
    rq3_model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO]
    
    rq3_model_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KU_ALL_XIN_ALL_AUTO : "KULTC+BAOLTC"
        }
    rq3_file_name = "rq3"
    generate_sk_data_for_remaining_ku_models(rq3_model_list, rq3_model_name, rq3_file_name)
    
    
    algo_model = ['rf', 'dtree', 'knn', 'NB', 'XGB', 'LGBM']
    ku_all_xin_all_algo_model_list = [KU_ALL_XIN_ALL_AUTO + '_hpt_' + m for m in algo_model]
    ku_cur_algo_model_list = [KU_CUR_XIN_AUTO + '_hpt_' + m for m in algo_model]
    
    rq4_model_list_1 = [KU_ALL_XIN_ALL_AUTO] + ku_all_xin_all_algo_model_list
    
    rq4_model_list_1_name = {
         KU_ALL_XIN_ALL_AUTO : "KULTC+BAOLTC",
        'ku_all_xin_all_auto_hpt_rf' : "KULTC+BAOLTC-HPT_RF",
        'ku_all_xin_all_auto_hpt_dtree': "KULTC+BAOLTC-HPT_DT",
        'ku_all_xin_all_auto_hpt_knn' : "KULTC+BAOLTC-HPT_KNN",
        'ku_all_xin_all_auto_hpt_NB' : "KULTC+BAOLTC-HPT_NB",
        'ku_all_xin_all_auto_hpt_XGB': "KULTC+BAOLTC-HPT_XGB",
        'ku_all_xin_all_auto_hpt_LGBM' : "KULTC+BAOLTC-HPT_LGBM"
        }
    rq4_file_name_1 = "rq4_f1"
    generate_sk_data_for_remaining_ku_models(rq4_model_list_1, rq4_model_list_1_name, rq4_file_name_1)
    
    
    rq5_model_list_1 = rq3_model_list + [KU_CUR_XIN_AUTO]  + ku_cur_algo_model_list
    rq5_model_list_1_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KU_ALL_XIN_ALL_AUTO : "KULTC+BAOLTC",
        'ku_present_proj_exp_xin_all_auto' : "KULTC-CUR+BAOLTC",
        'ku_present_proj_exp_xin_all_auto_hpt_rf' : "KULTC-CUR+BAOLTC-HPT_RF",
        'ku_present_proj_exp_xin_all_auto_hpt_dtree' : "KULTC-CUR+BAOLTC-HPT_DT",
        'ku_present_proj_exp_xin_all_auto_hpt_knn' : "KULTC-CUR+BAOLTC-HPT_KNN",
        'ku_present_proj_exp_xin_all_auto_hpt_NB' : "KULTC-CUR+BAOLTC-HPT_NB",
        'ku_present_proj_exp_xin_all_auto_hpt_XGB': "KULTC-CUR+BAOLTC-HPT_XGB",
        'ku_present_proj_exp_xin_all_auto_hpt_LGBM' : "KULTC-CUR+BAOLTC-HPT_LGBM"
         }
    
    rq5_file_name_1 = "rq5_f1"
    generate_sk_data_for_remaining_ku_models(rq5_model_list_1, rq5_model_list_1_name, rq5_file_name_1)
    
    
    ku_dev_expo_algo_model_list = [KU_DEV_EXP_XIN_AUTO + '_hpt_' + m for m in algo_model]
    
    rq5_model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO] + [KU_DEV_EXP_XIN_AUTO] # + ku_dev_expo_algo_model_list
    rq5_model_list_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KU_ALL_XIN_ALL_AUTO : "KULTC+BAOLTC",
        'ku_dev_exp_xin_all_auto' : "KULTC_DEV_EXP+BAOLTC"
        #'ku_dev_exp_xin_all_auto_hpt_rf' : "KULTC_DEV_EXP+BAOLTC-HPT_RF",
        #'ku_dev_exp_xin_all_auto_hpt_dtree' : "KULTC_DEV_EXP+BAOLTC-HPT_DT",
        #'ku_dev_exp_xin_all_auto_hpt_knn' : "KULTC_DEV_EXP+BAOLTC-HPT_KNN",
        #'ku_dev_exp_xin_all_auto_hpt_NB' : "KULTC_DEV_EXP+BAOLTC-HPT_NB",
        #'ku_dev_exp_xin_all_auto_hpt_XGB': "KULTC_DEV_EXP+BAOLTC-HPT_XGB",
        #'ku_dev_exp_xin_all_auto_hpt_LGBM' : "KULTC_DEV_EXP+BAOLTC-HPT_LGBM"
         }
    rq5_file_name = "rq5-f2"
    generate_sk_data_for_remaining_ku_models(rq5_model_list, rq5_model_list_name, rq5_file_name)

    
def rebuttal():
    # rebuttal
    rq6_model_list = [KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO] + [KU_DEV_EXP_XIN_AUTO, KU_DEV_EXP_AUTO] # + ku_dev_expo_algo_model_list
    rq6_model_list_name = {
        KU_ALL_AUTO: "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KU_ALL_XIN_ALL_AUTO : "KULTC+BAOLTC",
        'ku_dev_exp_xin_all_auto' : "KULTC_DEV_EXP+BAOLTC",
        KU_DEV_EXP_AUTO : 'KU_DEV_EXP'
        #'ku_dev_exp_xin_all_auto_hpt_rf' : "KULTC_DEV_EXP+BAOLTC-HPT_RF",
        #'ku_dev_exp_xin_all_auto_hpt_dtree' : "KULTC_DEV_EXP+BAOLTC-HPT_DT",
        #'ku_dev_exp_xin_all_auto_hpt_knn' : "KULTC_DEV_EXP+BAOLTC-HPT_KNN",
        #'ku_dev_exp_xin_all_auto_hpt_NB' : "KULTC_DEV_EXP+BAOLTC-HPT_NB",
        #'ku_dev_exp_xin_all_auto_hpt_XGB': "KULTC_DEV_EXP+BAOLTC-HPT_XGB",
        #'ku_dev_exp_xin_all_auto_hpt_LGBM' : "KULTC_DEV_EXP+BAOLTC-HPT_LGBM"
         }
    rq6_file_name = "rq6-f2"
    generate_sk_data_for_remaining_ku_models(rq6_model_list, rq6_model_list_name, rq6_file_name)

def rebuttal_model():
    rq1_model_list = [KULTC_SUM_AUTO, XIN_FEAT_AUTO]
    rq1_model_list_name = {
        KULTC_SUM_AUTO : "KULTC",
        XIN_FEAT_AUTO: "BAOLTC"
        }
    rq1_file_name = "reb-rq1"
    generate_sk_data_for_remaining_ku_models(rq1_model_list, rq1_model_list_name, rq1_file_name)

    
    rq2_model_list = [KULTC_SUM_AUTO, KU_ALL_XIN_ALL_AUTO, KULTC_SUM_XIN_AUTO]
    rq2_model_list_name = {
        KULTC_SUM_AUTO : "KULTC",
        XIN_FEAT_AUTO: "BAOLTC",
        KULTC_SUM_XIN_AUTO : "KULTC+BAOLTC",
        }
    rq2_file_name = "reb-rq2"
    generate_sk_data_for_remaining_ku_models(rq2_model_list, rq2_model_list_name, rq2_file_name)


if __name__ == "__main__":
    #main()
    #smote_hyper_data()
    #rebuttal()
    rebuttal_model()
    print("Program finishes successfully.")