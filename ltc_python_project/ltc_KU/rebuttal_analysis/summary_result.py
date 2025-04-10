#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:21:51 2025

@author: ahsan
"""

import pandas as pd
import numpy as np

f = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/results/full_model_result_jan_30_2025_med_sum.csv"

pd_data = pd.read_csv(f)

summary_result = pd_data.groupby(['model_name','year']).agg({'auc' : 'median'}).reset_index()

sort_summary_result = summary_result.sort_values(['year','model_name'], ascending=[True,False])



sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('xin_feature_auto','BAOLTC')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_dev_exp_xin_all_auto','KULTC_DEV_EXP+BAOLTC')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_sum_xin_all_auto','KULTC_SUM+BAOLTC')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_mid_xin_all_auto','KULTC_ORIGINAL(MED)+BAOLTC')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_mid_sum_xin_all_auto','KULTC_MED_SUM+BAOLTC')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_dim_sum_prof_auto','KULTC_SUM')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_dim_med_sum_prof_auto','KULTC_MED_SUM')
sort_summary_result['model_name'] = sort_summary_result['model_name'].str.replace('ku_all_dim_med_prof_auto','KULTC_ORIGINAL(MED)')

#sort_summary_result = sort_summary_result.sort_values(['year','model_name'], ascending=[True,False])

p = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/results/summary_result_models.csv'
sort_summary_result.to_csv(p,index=False)